"""CorrDiff-compatible UNet for EDM corrector training, with self-attention.

Call signature matches the EDMPrecond wrapper from the trainer:
    model(x_in, cond, c_noise, x_time_encode) -> (B, C_out, T, H, W)

Architectural notes:
  - 2D conv backbone with multi-head self-attention at low-resolution levels.
  - Time T is folded into batch (suits forecast_len=0 single-step training).
  - Noise label c_noise enters via a Fourier embedding -> MLP -> FiLM
    scale/shift on every ResBlock and AttnBlock.
  - Time-of-day encoding enters the same way.
  - Conditioning fields are concatenated to the input along the channel axis.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class FourierEmbedding(nn.Module):
    """Random-Fourier feature embedding for scalar inputs (noise level)."""

    def __init__(self, num_channels, scale=16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.float().ger((2 * math.pi * self.freqs).to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=-1)


class ConditionMLP(nn.Module):
    """Maps a feature vector to a per-block conditioning vector."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Residual block with FiLM conditioning from the (noise, time) embedding
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)

        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = (nn.Conv2d(in_ch, out_ch, kernel_size=1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


# --------------------------------------------------------------------------- #
# Multi-head self-attention block with optional FiLM conditioning
# --------------------------------------------------------------------------- #
class AttnBlock(nn.Module):
    """Multi-head self-attention over spatial dimensions."""

    def __init__(self, channels, emb_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # FiLM from the noise/time embedding.
        self.emb_proj = nn.Linear(emb_dim, 2 * channels)

    def forward(self, x, emb):
        B, C, H, W = x.shape
        h = self.norm(x)

        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = h * (1 + scale[..., None, None]) + shift[..., None, None]

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)

        # (B, heads, HW, head_dim) for scaled dot-product attention.
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)

        return x + self.dropout(self.proj(out))


# --------------------------------------------------------------------------- #
# Down / up sampling
# --------------------------------------------------------------------------- #
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# --------------------------------------------------------------------------- #
# UNet
# --------------------------------------------------------------------------- #
class CorrDiffUNet(nn.Module):
    """UNet denoiser for CorrDiff EDM corrector, with self-attention.

    Args:
        input_channel:     channels of x_in (the noisy residual).  Equals C_out.
        condition_channel: channels of cond (mu + static + forcing, concatenated).
        output_channel:    channels of predicted residual.  Equals input_channel.
        base_ch:           feature width at the highest resolution.
        ch_mults:          channel multipliers per resolution level.
        time_encode_dim:   x_time_encode input dim (e.g. 16 for 4 diurnal
                           harmonics + 4 annual harmonics).
        num_res_blocks:    residual blocks per resolution level.
        attn_levels:       which resolution levels (0-indexed, 0 = highest res)
                           get an AttnBlock after each ResBlock.  Default places
                           attention at the bottom two levels of a 5-level UNet.
        num_heads:         number of attention heads.  Must divide channels at
                           every attention level.
        dropout:           dropout rate in ResBlocks.
        fourier_dim:       size of the Fourier embedding for noise level.
    """

    def __init__(
        self,
        input_channel,
        condition_channel,
        output_channel,
        base_ch=128,
        ch_mults=(1, 2, 2, 4, 4),
        time_encode_dim=16,
        num_res_blocks=3,
        attn_levels=(3, 4),
        num_heads=8,
        dropout=0.13,
        fourier_dim=128,
        **kwargs,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.condition_channel = condition_channel
        self.output_channel = output_channel

        emb_dim = base_ch * 4
        n_levels = len(ch_mults)
        self._n_levels = n_levels
        self._num_res_blocks = num_res_blocks
        self.attn_levels = set(attn_levels)

        # ---- Embeddings ---- #
        self.noise_embed = nn.Sequential(
            FourierEmbedding(fourier_dim),
            ConditionMLP(fourier_dim, emb_dim),
        )
        self.time_embed = ConditionMLP(time_encode_dim, emb_dim)

        # ---- Input projection ---- #
        self.in_conv = nn.Conv2d(input_channel + condition_channel, base_ch,
                                 kernel_size=3, padding=1)

        # ---- Encoder ---- #
        # We use a flat ModuleList for down_blocks (mix of ResBlock and AttnBlock)
        # and a parallel structure list so the forward pass knows which is which.
        self.down_blocks = nn.ModuleList()
        self.down_block_types = []     # 'res' or 'attn'
        self.down_samples = nn.ModuleList()

        chs = [base_ch]
        ch = base_ch
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, emb_dim, dropout))
                self.down_block_types.append("res")
                ch = out_ch
                if i in self.attn_levels:
                    self.down_blocks.append(AttnBlock(ch, emb_dim, num_heads))
                    self.down_block_types.append("attn")
                chs.append(ch)        # skip captured after the (Res + optional Attn) pair
            if i < n_levels - 1:
                self.down_samples.append(Downsample(ch))
                chs.append(ch)
            else:
                self.down_samples.append(nn.Identity())

        # ---- Bottleneck: ResBlock - AttnBlock - ResBlock ---- #
        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttnBlock(ch, emb_dim, num_heads)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        # ---- Decoder (mirror of encoder) ---- #
        self.up_blocks = nn.ModuleList()
        self.up_block_types = []
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(ch_mults)):
            level_idx = n_levels - 1 - i      # original level index (matches encoder)
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, emb_dim, dropout))
                self.up_block_types.append("res")
                ch = out_ch
                if level_idx in self.attn_levels:
                    self.up_blocks.append(AttnBlock(ch, emb_dim, num_heads))
                    self.up_block_types.append("attn")
            if i < n_levels - 1:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(nn.Identity())

        # ---- Output head ---- #
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, output_channel, kernel_size=3, padding=1)

    # ------------------------------------------------------------------- #
    def forward(self, x_in, cond, c_noise, x_time_encode):
        """
        x_in:           (B, C_in,   T, H, W)   noisy residual
        cond:           (B, C_cond, T, H, W)   mu + static + forcing
        c_noise:        (B,)                   EDM noise label
        x_time_encode:  (B, D_time) or (B, T, D_time)
        Returns:        (B, C_out,  T, H, W)
        """
        B, C_in, T, H, W = x_in.shape

        # Fold time into batch so we can use 2D convs.
        x = x_in.permute(0, 2, 1, 3, 4).reshape(B * T, C_in, H, W)
        c = cond.permute(0, 2, 1, 3, 4).reshape(B * T, self.condition_channel, H, W)
        x = torch.cat([x, c], dim=1)

        # Build conditioning embedding (broadcast across T).
        emb_noise = self.noise_embed(c_noise)
        emb_noise = emb_noise.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        if x_time_encode.dim() == 2:
            x_time_encode = x_time_encode.unsqueeze(1).expand(B, T, -1)
        emb_time = self.time_embed(x_time_encode.reshape(B * T, -1).float())
        emb = emb_noise + emb_time

        # ---- Encoder ---- #
        h = self.in_conv(x)
        skips = [h]

        idx = 0
        for i in range(self._n_levels):
            for _ in range(self._num_res_blocks):
                # ResBlock
                h = self.down_blocks[idx](h, emb)
                idx += 1
                # Optional AttnBlock at this level
                if (idx < len(self.down_block_types)
                        and self.down_block_types[idx] == "attn"):
                    h = self.down_blocks[idx](h, emb)
                    idx += 1
                skips.append(h)
            h = self.down_samples[i](h)
            if not isinstance(self.down_samples[i], nn.Identity):
                skips.append(h)

        # ---- Bottleneck ---- #
        h = self.mid1(h, emb)
        h = self.mid_attn(h, emb)
        h = self.mid2(h, emb)

        # ---- Decoder ---- #
        idx = 0
        for i in range(self._n_levels):
            for _ in range(self._num_res_blocks + 1):
                # ResBlock with skip connection
                skip = skips.pop()
                h = self.up_blocks[idx](torch.cat([h, skip], dim=1), emb)
                idx += 1
                # Optional AttnBlock
                if (idx < len(self.up_block_types)
                        and self.up_block_types[idx] == "attn"):
                    h = self.up_blocks[idx](h, emb)
                    idx += 1
            if not isinstance(self.up_samples[i], nn.Identity):
                h = self.up_samples[i](h)

        # ---- Output ---- #
        h = self.out_conv(F.silu(self.out_norm(h)))

        # Unfold time back out.
        h = h.reshape(B, T, self.output_channel, H, W).permute(0, 2, 1, 3, 4)
        return h


    