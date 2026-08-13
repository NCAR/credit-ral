import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
import logging

from credit.postblock import PostBlock
from credit.models.base_model import BaseModel
from credit.boundary_padding import TensorPadding

logger = logging.getLogger(__name__)


def apply_spectral_norm(model):
    """
    Add spectral norm to all the conv and linear layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


def get_pad3d(input_resolution, window_size):
    """
    Estimate the size of padding based on the given window size and the original input size.

    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()

        # input size
        self.img_size = img_size

        # number of patches after embedding (T_num, Lat_num, Lon_num)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]
        self.patches_resolution = patches_resolution

        # number of embedded dimension after patching
        self.embed_dim = embed_dim

        # Conv3d-based patching
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # layer norm
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, T, C, Lat, Lon = x.shape

        # Conv3d-based patching and embedding
        x = self.proj(x)

        # combine T, Lat, Lon dimensions
        x = x.reshape(B, self.embed_dim, -1)

        # switch to channel-last for normalization
        x = x.transpose(1, 2)  # B T*Lat*Lon C

        # Layer norm (channel last)
        if self.norm is not None:
            x = self.norm(x)

        # switch back to channel first
        x = x.transpose(1, 2)

        # recover T, Lat, Lon dimensions
        x = x.reshape(B, self.embed_dim, *self.patches_resolution)

        return x

class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, num_residuals: int = 2):
        super().__init__()

        # Anti-aliased downsampling: stride-1 conv followed by average pooling
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # blocks of residual path
        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        # anti-aliased down-sampling
        x = self.conv(x)

        # skip-connection
        shortcut = x

        # residual path
        x = self.b(x)

        # additive residual connection
        return x + shortcut

class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()

        # Resize-convolution upsampling (no checkerboard artifacts)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
        )

        # blocks of residual path
        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        # up-sampling (bilinear + conv)
        x = self.up(x)

        # skip-connection
        shortcut = x

        # residual path
        x = self.b(x)

        # additive residual connection
        return x + shortcut


class UTransformer(nn.Module):
    """U-Transformer

    Args:
        embed_dim (int): Patch embedding dimension.
        num_groups (int | tuple[int]): number of groups to separate the channels into.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """

    def __init__(
        self,
        embed_dim,
        num_groups,
        input_resolution,
        num_heads,
        window_size,
        depth,
        drop_path,
    ):
        super().__init__()
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)  # convert window_size[int] to tuple

        # padding input tensors so they are divided by the window size
        padding = get_pad2d(input_resolution, window_size)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding

        # -------------------------------------------------------------------
        # FIX 2: Reflection padding instead of ZeroPad2d
        #   - ZeroPad2d injects artificial zeros at boundaries, causing
        #     tokens near edges to attend to meaningless values.
        #   - Reflection padding mirrors the real spatial content, producing
        #     physically meaningful boundary representations.
        # -------------------------------------------------------------------
        self.pad_values = (padding_left, padding_right, padding_top, padding_bottom)

        # input resolution after padding
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right

        # down-sampling block (now anti-aliased)
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])

        # SwinT block
        self.layer = SwinTransformerV2Stage(
            embed_dim,
            embed_dim,
            input_resolution,
            depth,
            num_heads,
            window_size[0],
            drop_path=drop_path,
        )

        # up-sampling block (now upsample + conv)
        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding

        x = self.down(x)
        shortcut = x

        # print(f'U-Transform pad value: {self.pad_values}')
        # FIX 2: Reflection padding instead of zero padding
        # x = F.pad(x, self.pad_values, mode="reflect")
        x = F.pad(x, self.pad_values, mode="replicate")
        _, _, pad_lat, pad_lon = x.shape

        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        # crop back to original (pre-pad) spatial size
        x = x[
            :,
            :,
            padding_top: pad_lat - padding_bottom,
            padding_left: pad_lon - padding_right,
        ]

        # concat skip connection
        x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon
        x = self.up(x)
        return x


# ---------------------------------------------------------------------------
# FIX 1: Convolutional Decoder with PixelShuffle (replaces fc + reshape)
#
#   The original code used:
#       self.fc = nn.Linear(dim, out_chans * patch_h * patch_w)
#   followed by a reshape to tile patches edge-to-edge. Each patch was
#   independently reconstructed from a single token with NO spatial
#   communication between neighbors -- the direct cause of tile seams.
#
#   This decoder uses Conv2d (3x3 kernel) so each output location is
#   influenced by a neighborhood of tokens, then PixelShuffle to
#   upsample to full resolution. This eliminates seam artifacts.
# ---------------------------------------------------------------------------
class ConvPixelShuffleDecoder(nn.Module):
    """
    Convolutional decoder that replaces the fc + reshape unpatchify.
    Uses sub-pixel convolution (PixelShuffle) to avoid tile artifacts.

    Args:
        embed_dim (int): Input channel dimension (from transformer).
        out_chans (int): Number of output channels (meteorological variables).
        upscale_h (int): Patch height (spatial upscale factor in lat).
        upscale_w (int): Patch width (spatial upscale factor in lon).
        num_groups (int): Groups for GroupNorm.
    """

    def __init__(self, embed_dim, out_chans, upscale_h, upscale_w, num_groups=32):
        super().__init__()

        self.upscale_h = upscale_h
        self.upscale_w = upscale_w

        if upscale_h == upscale_w:
            # Symmetric patch size -- standard PixelShuffle
            upscale = upscale_h
            shuffle_chans = out_chans * (upscale ** 2)
            self.decoder = nn.Sequential(
                # 3x3 conv: each output location sees a neighborhood of tokens
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, embed_dim),
                nn.SiLU(),
                # Second conv for more capacity
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, embed_dim),
                nn.SiLU(),
                # Project to shuffle-ready channels
                nn.Conv2d(embed_dim, shuffle_chans, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale),
            )
            self.use_symmetric = True
        else:
            # Asymmetric patch size -- manual sub-pixel rearrangement
            shuffle_chans = out_chans * upscale_h * upscale_w
            self.decoder = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, embed_dim),
                nn.SiLU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, embed_dim),
                nn.SiLU(),
                nn.Conv2d(embed_dim, shuffle_chans, kernel_size=3, padding=1),
            )
            self.out_chans = out_chans
            self.use_symmetric = False

    def forward(self, x):
        """
        Args:
            x: (B, embed_dim, Lat_tokens, Lon_tokens)
        Returns:
            (B, out_chans, Lat_tokens * upscale_h, Lon_tokens * upscale_w)
        """
        x = self.decoder(x)

        if not self.use_symmetric:
            # Manual sub-pixel rearrangement for asymmetric patch sizes
            B, C, H, W = x.shape
            x = x.reshape(B, self.out_chans, self.upscale_h, self.upscale_w, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3)  # B, C, H, uh, W, uw
            x = x.reshape(B, self.out_chans, H * self.upscale_h, W * self.upscale_w)

        return x


# ---------------------------------------------------------------------------
# FIX 5: Gated fusion module (replaces simple addition x + x_outside)
#
#   Simple addition doesn't let the model learn WHERE to trust inside vs.
#   outside information. A learned sigmoid gate allows spatial modulation:
#   mountains/coasts can weight outside context differently from flat terrain.
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learned spatial gating for fusing inside and outside embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x_inside, x_outside):
        """
        Args:
            x_inside:  (B, dim, H, W) -- high-res interior embedding
            x_outside: (B, dim, H, W) -- coarse exterior embedding
        Returns:
            (B, dim, H, W) -- gated combination
        """
        gate = self.gate(torch.cat([x_inside, x_outside], dim=1))
        return gate * x_inside + (1 - gate) * x_outside


class WRF_Tansformer(BaseModel):
    """
    Fixed WRF Transformer with:
      1. ConvPixelShuffleDecoder replacing fc + reshape (fixes tile seams)
      2. Reflection padding replacing ZeroPad2d (fixes edge artifacts)
      3. Anti-aliased DownBlock (fixes shift-variance)
      4. Upsample+Conv UpBlock (fixes checkerboard from ConvTranspose2d)
      5. GatedFusion replacing additive merge (better inside/outside blend)

    Args:
        img_size (Sequence[int], optional): T, Lat, Lon.
        patch_size (Sequence[int], optional): T, Lat, Lon.
        in_chans (int, optional): number of input channels.
        out_chans (int, optional): number of output channels.
        dim (int, optional): number of embed channels.
        num_groups (Sequence[int] | int, optional): number of groups to separate the channels into.
        num_heads (int, optional): Number of attention heads.
        window_size (int | tuple[int], optional): Local window size.
    """

    def __init__(
        self,
        param_interior,
        param_outside,
        time_encode_dim=12,
        num_groups=32,
        num_heads=8,
        depth=48,
        window_size=7,
        use_spectral_norm=True,
        interp=True,
        drop_path=0,
        padding_conf=None,
        post_conf=None,
        **kwargs,
    ):
        super().__init__()
        self.time_encode = time_encode_dim
        image_height_inside = param_interior["image_height"]
        patch_height_inside = param_interior["patch_height"]
        image_width_inside = param_interior["image_width"]
        patch_width_inside = param_interior["patch_width"]
        levels_inside = param_interior["levels"]
        frames_inside = param_interior["frames"]
        frame_patch_size_inside = param_interior["frame_patch_size"]
        channels_inside = param_interior["channels"]
        surface_channels_inside = param_interior["surface_channels"]
        input_only_channels_inside = param_interior["input_only_channels"]
        output_only_channels_inside = param_interior["output_only_channels"]
        dim_inside = param_interior["dim"]

        image_height_outside = param_outside["image_height"]
        patch_height_outside = param_outside["patch_height"]
        image_width_outside = param_outside["image_width"]
        patch_width_outside = param_outside["patch_width"]
        levels_outside = param_outside["levels"]
        frames_outside = param_outside["frames"]
        frame_patch_size_outside = param_outside["frame_patch_size"]
        channels_outside = param_outside["channels"]
        surface_channels_outside = param_outside["surface_channels"]
        dim_outside = param_outside["dim"]

        self.use_interp = interp
        self.use_spectral_norm = use_spectral_norm

        if padding_conf is None:
            padding_conf = {"activate": False}

        self.use_padding = padding_conf["activate"]

        if post_conf is None:
            post_conf = {"activate": False}

        self.use_post_block = post_conf["activate"]

        # input tensor size (time, lat, lon)
        if self.use_padding:
            pad_lat = padding_conf["pad_lat"]
            pad_lon = padding_conf["pad_lon"]
            image_height_pad = image_height_inside + pad_lat[0] + pad_lat[1]
            image_width_pad = image_width_inside + pad_lon[0] + pad_lon[1]

            img_size_inside = (frames_inside, image_height_pad, image_width_pad)
            self.img_size_original = (
                frames_inside,
                image_height_inside,
                image_width_inside,
            )
        else:
            img_size_inside = (frames_inside, image_height_inside, image_width_inside)
            self.img_size_original = img_size_inside

        img_size_outside = (frames_outside, image_height_outside, image_width_outside)

        # the size of embedded patches
        patch_size_inside = (
            frame_patch_size_inside,
            patch_height_inside,
            patch_width_inside,
        )
        patch_size_outside = (
            frame_patch_size_outside,
            patch_height_outside,
            patch_width_outside,
        )

        # number of channels = levels * variables per level + surface variables
        in_chans_inside = channels_inside * levels_inside + surface_channels_inside + input_only_channels_inside
        out_chans_inside = channels_inside * levels_inside + surface_channels_inside + output_only_channels_inside

        in_chans_outside = channels_outside * levels_outside + surface_channels_outside

        # input resolution = number of embedded patches / 2
        # divide by two because "u_transformer" has a down-sampling block
        input_resolution_inside = (
            round(img_size_inside[1] / patch_size_inside[1] / 2),
            round(img_size_inside[2] / patch_size_inside[2] / 2),
        )

        input_resolution_outside = (
            round(img_size_outside[1] / patch_size_outside[1] / 2),
            round(img_size_outside[2] / patch_size_outside[2] / 2),
        )

        # FuXi cube embedding layer
        self.cube_embedding_inside = CubeEmbedding(
            img_size_inside, patch_size_inside, in_chans_inside, dim_inside
        )
        self.cube_embedding_outside = CubeEmbedding(
            img_size_outside, patch_size_outside, in_chans_outside, dim_outside
        )
        self.total_dim = dim_inside

        # Downsampling --> SwinTransformerV2 stacks --> Upsampling
        self.u_transformer = UTransformer(
            self.total_dim,
            num_groups,
            input_resolution_inside,
            num_heads,
            window_size,
            depth=depth,
            drop_path=drop_path,
        )

        # -------------------------------------------------------------------
        # FIX 1: ConvPixelShuffleDecoder replaces fc + reshape
        # -------------------------------------------------------------------
        _, patch_lat, patch_lon = patch_size_inside
        self.decoder = ConvPixelShuffleDecoder(
            embed_dim=self.total_dim,
            out_chans=out_chans_inside,
            upscale_h=patch_lat,
            upscale_w=patch_lon,
            num_groups=min(num_groups, self.total_dim),
        )

        # -------------------------------------------------------------------
        # FIX 5: GatedFusion replaces x = x + x_outside
        # -------------------------------------------------------------------
        self.gated_fusion = GatedFusion(self.total_dim)

        # Hyperparameters
        self.patch_size = patch_size_inside
        self.input_resolution = input_resolution_inside
        self.out_chans = out_chans_inside
        self.img_size = img_size_inside

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            apply_spectral_norm(self)

        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

        if self.time_encode > 0:
            self.film = nn.Linear(self.time_encode, 2 * (self.total_dim))

    def _match_spatial(self, src: torch.Tensor, ref: torch.Tensor):
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        x_outside: torch.Tensor,
        x_extra: torch.Tensor,
    ):
        # copy tensor to feed into postblock later
        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        # Tensor dims: Batch, Variables, Time, Lat grids, Lon grids
        B, _, _, _, _ = x.shape

        _, patch_lat, patch_lon = self.patch_size

        # Get the number of patches after embedding
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2

        # Cube Embedding and squeeze the time dimension
        x = self.cube_embedding_inside(x).squeeze(2)  # B C Lat Lon

        x_outside = self.cube_embedding_outside(x_outside).squeeze(2)  # B C Lat Lon

        if self.time_encode > 0:
            # Feature-wise Linear Modulation
            alpha_beta = self.film(x_extra)  # [batch, 2*dim]
            alpha, beta = alpha_beta.chunk(2, dim=1)  # each is [batch, dim]
            alpha = alpha.view(B, self.total_dim, 1, 1)
            beta = beta.view(B, self.total_dim, 1, 1)
            x_outside = alpha * x_outside + beta

        # -------------------------------------------------------------------
        # FIX 5: Gated fusion instead of simple addition
        # -------------------------------------------------------------------
        x = self.gated_fusion(x, x_outside)

        # U-Transformer stage
        x = self.u_transformer(x)

        # -------------------------------------------------------------------
        # FIX 1: Convolutional decoder with PixelShuffle
        #   - Replaces: self.fc(x.permute(...)) + reshape
        #   - Each output pixel now depends on a 3x3 neighborhood of tokens,
        #     eliminating the tile seam artifacts.
        # -------------------------------------------------------------------
        x = self.decoder(x)  # B, out_chans, Lat*patch_lat, Lon*patch_lon

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            img_size = list(self.img_size_original)
            x = F.interpolate(x, size=img_size[1:], mode="bilinear")

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x
        