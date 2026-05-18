import torch
import torch.nn as nn
import torch.nn.functional as F


class GateGenerator(nn.Module):
    """Conv-based gate: feature → [0,1] mask."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PixelRouter(nn.Module):
    """Pixel-wise router: concat(spa, spe) → 3-path logits."""
    def __init__(self, dim, hidden_dim=None, num_paths=3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(dim // 4, 8)
        self.router = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_paths, kernel_size=1, bias=True),
        )

    def forward(self, f_spa, f_spe):
        router_input = torch.cat([f_spa, f_spe], dim=1)        # [B, 2C, H, W]
        route_logits = self.router(router_input)               # [B, 3, H, W]
        route_weights = torch.softmax(route_logits, dim=1)    # [B, 3, H, W]
        return route_weights


class DynamicFusionRouter(nn.Module):
    """
    V4 Dynamic Fusion Router.

    Combines 3 fusion paths via pixel-wise learned weights:
      Path 1 — Baseline:        f1 = spa + spe
      Path 2 — S2S (spa→spe):   g_spa = spa_gate(spa), spe' = spe + α_s2s * g_spa * spe
      Path 3 — S2P (spe→spa):   g_spe = spe_gate(spe), spa' = spa + α_s2p * g_spe * spa

    Pixel Router: concat(spa, spe) → Conv1x1 → BN → GELU → Conv1x1 → 3-ch softmax
    Final output: f_fuse = w1*f1 + w2*f2 + w3*f3

    Input:  f_spa [B, C, H, W], f_spe [B, C, H, W]
    Output: f_fuse [B, C, H, W], route_weights [B, 3, H, W]
    """
    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(channels // 4, 8)

        # Gate generators (spectral-only or spatial-only input)
        self.spa_gate = GateGenerator(channels, hidden_dim)   # Path 2: spa → modulate spe
        self.spe_gate = GateGenerator(channels, hidden_dim)   # Path 3: spe → modulate spa

        # Pixel-wise router
        self.pixel_router = PixelRouter(channels, hidden_dim, num_paths=3)

        # Learnable residual scaling (init 0 = identity / pure add)
        self.alpha_s2s = nn.Parameter(torch.zeros(1))
        self.alpha_s2p = nn.Parameter(torch.zeros(1))

        # Store last route weights for logging
        self.last_route_weights = None

    def forward(self, f_spa, f_spe):
        # --- Shape checks ---
        assert f_spa.dim() == 4, f"Expected 4D input, got {f_spa.dim()}D"
        assert f_spa.shape == f_spe.shape, (
            f"Shape mismatch: f_spa {list(f_spa.shape)} vs f_spe {list(f_spe.shape)}"
        )

        # --- Path 1: Baseline add ---
        f1 = f_spa + f_spe                                                  # [B, C, H, W]

        # --- Path 2: S2S (spa → modulate spe) ---
        g_spa = self.spa_gate(f_spa)                                        # [B, C, H, W]
        spe_guided = f_spe + self.alpha_s2s * g_spa * f_spe
        f2 = f_spa + spe_guided                                             # [B, C, H, W]

        # --- Path 3: S2P (spe → modulate spa) ---
        g_spe = self.spe_gate(f_spe)                                        # [B, C, H, W]
        spa_guided = f_spa + self.alpha_s2p * g_spe * f_spa
        f3 = spa_guided + f_spe                                             # [B, C, H, W]

        # --- Pixel-wise routing ---
        route_weights = self.pixel_router(f_spa, f_spe)                     # [B, 3, H, W]
        w1 = route_weights[:, 0:1, :, :]                                    # [B, 1, H, W]
        w2 = route_weights[:, 1:2, :, :]
        w3 = route_weights[:, 2:3, :, :]

        f_fuse = w1 * f1 + w2 * f2 + w3 * f3                               # [B, C, H, W]

        self.last_route_weights = route_weights.detach()
        return f_fuse, route_weights
