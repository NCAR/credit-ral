import torch
from torch import nn
from torch.nn import functional as F
import logging

from credit.postblock import PostBlock
from credit.models.base_model import BaseModel
from credit.boundary_padding import TensorPadding

logger = logging.getLogger(__name__)

def apply_spectral_norm(model):
    """
    add spectral norm to all the conv and linear layers
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)

# -------------------- #
# U-Net building blocks
# -------------------- #
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2 with same-padding so H/W are preserved."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale by 2 (MaxPool) then DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """
    Upscale by 2 (ConvTranspose2d) then concat skip, then DoubleConv.
    Includes padding logic to handle odd spatial sizes.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle mismatched shapes (common when input H/W not divisible by 2^depth)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)




class Dscale_UNET(BaseModel):
    
    def __init__(
        self,
        image_height=640,
        image_width=1280,
        total_input_channels=50,
        total_target_channels=80,
        dim=[64, 128, 256, 512],
        use_spectral_norm=True,
        padding_conf=None,
        post_conf=None,
        interp=False,
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

        # input tensor size notion (time, lat, lon) — only lat/lon used for interpolate
        if self.use_padding:
            pad_lat = padding_conf["pad_lat"]
            pad_lon = padding_conf["pad_lon"]
            image_height_pad = image_height + pad_lat[0] + pad_lat[1]
            image_width_pad = image_width + pad_lon[0] + pad_lon[1]
            self.img_size_original = (frames, image_height, image_width)
            self.img_size_pad = (frames, image_height_pad, image_width_pad)
        else:
            self.img_size_original = (frames, image_height, image_width)
            self.img_size_pad = self.img_size_original

        self.in_chans = int(total_input_channels)
        self.out_chans = int(total_target_channels)

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        if len(dim) < 2:
            raise ValueError("dim must have at least 2 stages (e.g. [64,128,256,512]).")
        dim = list(dim)

        # --------------------------- #
        # U-Net model hyperparameters
        # --------------------------- #
        # Encoder
        self.inc = DoubleConv(self.in_chans, dim[0])
        self.downs = nn.ModuleList(
            [Down(dim[i], dim[i + 1]) for i in range(len(dim) - 1)]
        )

        # Bottleneck (one more downsample)
        self.bottleneck = Down(dim[-1], dim[-1] * 2)

        # Decoder
        rev = list(reversed(dim))  # e.g. [512,256,128,64]
        up_in = dim[-1] * 2        # bottleneck channels
        self.ups = nn.ModuleList()
        for out_ch in rev:
            self.ups.append(Up(up_in, out_ch))
            up_in = out_ch

        # Output head
        self.outc = nn.Conv2d(dim[0], self.out_chans, kernel_size=1)

        # Optional post block (create BEFORE moving to device)
        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

        # Optional spectral norm
        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        # Move the model to the device (kept from your script, but done at the end)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x: torch.Tensor):
        # Copy tensor to feed into postblock later (keeps your behavior)
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
        
        # -------------------- #
        # U-Net forward
        # -------------------- #
        # Encoder + skips
        x1 = self.inc(x)
        skips = [x1]
        x_enc = x1
        for down in self.downs:
            x_enc = down(x_enc)
            skips.append(x_enc)

        # Bottleneck
        x_dec = self.bottleneck(x_enc)

        # Decoder (consume skips in reverse: last skip first)
        for up, skip in zip(self.ups, reversed(skips)):
            x_dec = up(x_dec, skip)

        # Output projection
        x = self.outc(x_dec)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        # Optional final resize to original lat/lon
        if self.use_interp:
            img_size = list(self.img_size_original)
            x = F.interpolate(x, size=img_size[1:], mode="bilinear", align_corners=False)

        # Optional post-processing block
        if self.use_post_block:
            x = {"y_pred": x, "x": x_copy}
            x = self.postblock(x)

        return x
        

