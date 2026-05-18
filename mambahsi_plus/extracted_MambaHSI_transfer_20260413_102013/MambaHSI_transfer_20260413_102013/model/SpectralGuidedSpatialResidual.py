import torch
import torch.nn as nn


class SpectralGuidedSpatialResidual(nn.Module):
    """
    V5: Spectral-Guided Spatial Residual.

    Core idea — spectral features only modulate the spatial RESIDUAL, not the
    spatial feature directly.  This keeps the Baseline (spa + spe) stable and
    lets the model learn how much spectral-guided correction to inject.

        res_spa   = spa_residual(f_spa)
        gate      = spe_gate(f_spe)
        f_spa_ref = f_spa + alpha * gate * res_spa
        f_fuse    = f_spa_ref + f_spe

    alpha is initialised to 0 so training starts identically to Baseline.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 8)

        self.spa_residual = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
        )

        self.spe_gate = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_spa, f_spe):
        assert f_spa.dim() == 4, f"Expected 4D input, got {f_spa.dim()}D"
        assert f_spa.shape == f_spe.shape, (
            f"Shape mismatch: f_spa {list(f_spa.shape)} vs f_spe {list(f_spe.shape)}"
        )

        res_spa = self.spa_residual(f_spa)            # [B, C, H, W]
        gate = self.spe_gate(f_spe)                   # [B, C, H, W] in [0,1]

        assert res_spa.shape == f_spa.shape
        assert gate.shape == f_spa.shape

        f_spa_refined = f_spa + self.alpha * gate * res_spa
        f_fuse = f_spa_refined + f_spe

        assert f_fuse.shape == f_spa.shape
        return f_fuse
