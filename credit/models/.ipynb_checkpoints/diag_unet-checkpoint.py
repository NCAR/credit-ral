import torch
from torch import nn
from torch.nn import functional as F
import logging

from credit.postblock import PostBlock
from credit.models.base_model import BaseModel
from credit.boundary_padding import TensorPadding

logger = logging.getLogger(__name__)


def apply_spectral_norm(model: nn.Module) -> None:
    """
    Add spectral norm to all Conv/ConvTranspose/Linear layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


# ----------------------------
# U-Net building blocks
# ----------------------------
class DoubleConv(nn.Module):
    """(conv => norm => ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "group", groups: int = 8):
        super().__init__()

        def _norm_layer(ch: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm2d(ch)
            if norm == "instance":
                return nn.InstanceNorm2d(ch, affine=True)
            if norm == "group":
                g = min(groups, ch)
                # make g divide ch
                while g > 1 and (ch % g) != 0:
                    g -= 1
                return nn.GroupNorm(g, ch)
            return nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale with maxpool then double conv"""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """
    Upscale then concat skip connection then double conv.

    If bilinear=True, uses interpolation.
    If bilinear=False, uses ConvTranspose2d.
    """

    def __init__(self, x_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True, norm: str = "group"):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in = x_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(x_ch, out_ch, kernel_size=2, stride=2)
            conv_in = out_ch + skip_ch

        self.conv = DoubleConv(conv_in, out_ch, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # If spatial sizes are off by 1 due to odd dims, pad x to match skip.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ----------------------------
# Main model
# ----------------------------
class Diag_UNET(BaseModel):
    def __init__(
        self,
        image_height=640,
        image_width=1280,
        total_input_channels=50,
        total_target_channels=80,
        time_encode_dim=4,
        dim=(64, 128, 256, 512),
        frames=2,
        use_spectral_norm=True,
        padding_conf=None,
        post_conf=None,
        interp=True,  # also controls U-Net decoder upsampling mode
        norm="group",
        **kwargs,
    ):
        super().__init__()

        self.use_interp = bool(interp)
        self.use_spectral_norm = bool(use_spectral_norm)

        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = bool(padding_conf.get("activate", False))

        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = bool(post_conf.get("activate", False))

        # input tensor size (time, lat, lon)
        if self.use_padding:
            pad_lat = padding_conf["pad_lat"]
            pad_lon = padding_conf["pad_lon"]
            image_height_pad = image_height + pad_lat[0] + pad_lat[1]
            image_width_pad = image_width + pad_lon[0] + pad_lon[1]
            img_size = (frames, image_height_pad, image_width_pad)
            self.img_size_original = (frames, image_height, image_width)
        else:
            img_size = (frames, image_height, image_width)
            self.img_size_original = img_size

        in_chans = int(total_input_channels)
        out_chans = int(total_target_channels)

        dims = list(dim) if isinstance(dim, (list, tuple)) else [int(dim)]
        if len(dims) < 2:
            raise ValueError("`dim` must have at least 2 channel sizes, e.g. (64, 128, 256, 512).")

        self.frames = int(frames)
        self.out_chans = out_chans
        self.img_size = img_size
        self.input_resolution = tuple(img_size[1:])  # (H, W)

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # ----------------------------------------
        # U-Net: squeeze frames into channels
        # ----------------------------------------
        self.inc = DoubleConv(in_chans * self.frames, dims[0], norm=norm)
        self.downs = nn.ModuleList([Down(dims[i], dims[i + 1], norm=norm) for i in range(len(dims) - 1)])
        self.ups = nn.ModuleList(
            [
                Up(
                    x_ch=dims[i],
                    skip_ch=dims[i - 1],
                    out_ch=dims[i - 1],
                    bilinear=self.use_interp,
                    norm=norm,
                )
                for i in range(len(dims) - 1, 0, -1)
            ]
        )

        # 1x1 conv head (equivalent to "dense on channels per pixel")
        self.fc = nn.Conv2d(dims[0], out_chans, kernel_size=1)

        # ----------------------------------------
        # FiLM conditioning (applied at dims[0])
        # ----------------------------------------
        self.total_dim = dims[0]
        self.time_encode = int(time_encode_dim)
        self.film = nn.Linear(self.time_encode, 2 * self.total_dim)

        # Move the model to the device (kept to match your original pattern)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

    def forward(self, x: torch.Tensor, x_extra: torch.Tensor):
        # copy tensor to feed into postblock later
        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        B, C, T, H, W = x.shape

        # ======================================== #
        # UNET block: embed and squeeze frames dim
        # ======================================== #
        # squeeze frames into channels: (B, C*T, H, W)
        x = x.reshape(B, C * T, H, W)
        x = self.inc(x)  # (B, dim0, H, W)

        # =========================================== #
        # Feature‑wise Linear Modulation for x_extra
        x_extra = x_extra.to(x.device)
        alpha_beta = self.film(x_extra)  # [batch, 2*dim0]
        alpha, beta = alpha_beta.chunk(2, dim=1)  # each is [batch, dim0]
        alpha = alpha.view(B, self.total_dim, 1, 1)  # [batch, dim0, 1, 1]
        beta = beta.view(B, self.total_dim, 1, 1)  # [batch, dim0, 1, 1]
        x = alpha * x + beta

        # ======================================== #
        # UNET main blocks; x is output
        # ======================================== #
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # last element is bottleneck feature; pop it then decode with remaining skips
        x = skips.pop()
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        x = self.fc(x)  # (B, out_chans, H, W)

        if self.use_padding:
            # TensorPadding implementations sometimes expect 4D or 5D;
            # try 4D, fallback to 5D with a dummy time dim.
            try:
                x = self.padding_opt.unpad(x)
            except Exception:
                x = self.padding_opt.unpad(x.unsqueeze(2)).squeeze(2)

        if self.use_interp:
            img_size = list(self.img_size_original)  # (frames, H, W)
            x = F.interpolate(x, size=img_size[1:], mode="bilinear", align_corners=False)

        # restore a singleton frames dim for downstream code: (B, out_chans, 1, H, W)
        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x

