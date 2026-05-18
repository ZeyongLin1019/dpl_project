import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math

def padding_feature(self, x):
    """Ensure the input feature map has the required number of channels."""
    B, C, H, W = x.shape
    if C < self.channel_num:
        pad_c = self.channel_num - C
        pad_features = torch.zeros((B, pad_c, H, W), device=x.device)
        return torch.cat([x, pad_features], dim=1)
    return x


class ECALayer(nn.Module):
    """
    ECA: Efficient Channel Attention with Mamba
    """
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mamba = Mamba(d_model=channel, d_state=16, d_conv=4, expand=2) 

    def forward(self, x):
        # x: [B, C, H, W]
        # x = self.conv(x)
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).permute(0, 2, 1)  # [B, C, 1] -> [B, 1, C]

        y = self.mamba(y)

        y = y.permute(0, 2, 1).unsqueeze(-1)  # [B, 1, C] -> [B, C, 1] -> [B, C, 1, 1]
        return x * torch.sigmoid(y)

class SpaMambaProcessor(nn.Module):
    def __init__(self, channels):
        super(SpaMambaProcessor, self).__init__()
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        """
        Applies Mamba processing along one spatial dimension.
        x: B H W C
        """
        B, H, W, C = x.shape
        x_flat = x.view(B, -1, x.shape[-1])  # Flatten to [B, H*W, C]
        x_proc = self.mamba(x_flat)

        x_flipped = torch.flip(x_flat, dims=[1])
        x_proc_flipped = torch.flip(self.mamba(x_flipped), dims=[1])
        return x_proc.view(*x.shape) + x_proc_flipped.view(*x.shape)


class SpeMambaProcessor(nn.Module):
    def __init__(self, group_channel_num):
        super(SpeMambaProcessor, self).__init__()
        self.mamba = Mamba(d_model=group_channel_num, d_state=16, d_conv=4, expand=2)
        self.group_channel_num = group_channel_num

    def forward(self, x):
        B, H, W, C = x.shape
        x_flat = x.view(B * H * W, -1, self.group_channel_num)  # Flatten to [1, B*H*W, C]
        # x_flat = x.view(B * H * W, -1, 128)  # Flatten to [1, B*H*W, C]
        x_flipped = torch.flip(x_flat, dims=[1])
        x_proc = self.mamba(x_flat)
        x_proc_flipped = torch.flip(self.mamba(x_flipped), dims=[1])
        return x_proc.view(*x.shape) + x_proc_flipped.view(*x.shape)

class SpatialGuidedSpectralFusion(nn.Module):
    """
    S2S Fusion V2: dual-branch control gate.

    Control weights are generated from concatenated spatial and spectral
    features, then modulate the spectral branch output via a learnable alpha.

    Input:  f_spa [B, C, H, W], f_spe [B, C, H, W]
    Output: f_spe_guided [B, C, H, W]
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_dim = max(1, channels // reduction)
        self.control = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_spa, f_spe):
        if f_spa.dim() != 4:
            raise ValueError(f"Expected 4D input, got {f_spa.dim()}D")
        if f_spa.shape != f_spe.shape:
            raise ValueError(
                f"Shape mismatch: f_spa {list(f_spa.shape)} vs f_spe {list(f_spe.shape)}"
            )
        control_input = torch.cat([f_spa, f_spe], dim=1)      # [B, 2C, H, W]
        c = self.control(control_input)                       # [B, C, H, W] in [0,1]
        f_spe_guided = f_spe + self.alpha * c * f_spe
        f_spe_guided = self.out_proj(f_spe_guided)
        return f_spe_guided


class SpectralGuidedSpatialFusion(nn.Module):
    """
    Spectral-to-Spatial Fusion (S2P): spectral features gate spatial features.

    Uses spectral branch features to generate control weights that modulate
    the spatial branch output via a learnable alpha (init 0 = identity).

    Input:  f_spa [B, C, H, W], f_spe [B, C, H, W]
    Output: f_spa_guided [B, C, H, W]
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_dim = max(1, channels // reduction)
        self.control = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_spa, f_spe):
        if f_spa.dim() != 4:
            raise ValueError(f"Expected 4D input, got {f_spa.dim()}D")
        if f_spa.shape != f_spe.shape:
            raise ValueError(
                f"Shape mismatch: f_spa {list(f_spa.shape)} vs f_spe {list(f_spe.shape)}"
            )
        gate = self.control(f_spe)      # [B, C, H, W] in [0,1], spectral→spatial
        f_spa_guided = f_spa + self.alpha * gate * f_spa
        f_spa_guided = self.out_proj(f_spa_guided)
        return f_spa_guided


class CustomAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        B, L, D = x.shape
        residual = x

        x = self.layer_norm(x)

        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) 
                   for t in qkv]  # (B, H, L, C/H)

        k_t = k.transpose(-2, -1)  # (B, H, C/H, L)
        kt_v = torch.matmul(k_t, v)  # (B, H, C/H, C/H)
        softmax_kt_v = F.softmax(kt_v, dim=-1) 
        output = torch.matmul(q, softmax_kt_v)

        output = output.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        output = self.out_proj(output)

        output = self.dropout(output)
        return output + residual
