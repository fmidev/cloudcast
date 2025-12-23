import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2
from swinu.util import radial_bins_rfft, apply_hann_window


class BSPLoss(nn.Module):
    """
    Binned Spectral Power (BSP) loss + pixel MSE.

    L_total = MSE_pixel + lambda_bsp * L_BSP

    https://arxiv.org/pdf/2502.00472

    Shapes supported:
      [B, C, H, W]  -> internally treated as [B, 1, C, H, W]
      [B, T, C, H, W]
    """

    def __init__(
        self,
        lambda_bsp: float = 1.0,  # weight for BSP component
        n_bins: int | None = None,
        eps: float = 1e-8,  # epsilon in energy ratio
        gamma: float = 0.0,  # exponent for bin weights
        kmax_frac: float = 0.707,  # remove all bins below nyquist (assuming 5km spacing)
    ):
        super().__init__()
        self.lambda_bsp = lambda_bsp
        self.n_bins = n_bins
        self.eps = eps
        self.gamma = gamma
        self.kmax_frac = kmax_frac

        self.k = None  # will be filled once we know n_bins

    def _diag(self, E_pred, E_true, device):
        """
        Simple diagnostics over low/mid/high bands.
        E_pred, E_true: [N, n_bins]
        """
        low = self.k < 0.20
        mid = (self.k >= 0.20) & (self.k < 0.45)
        high = self.k >= 0.45

        r = (E_pred + self.eps) / (E_true + self.eps)
        lpe = torch.log(E_pred + self.eps) - torch.log(E_true + self.eps)

        def band_mean(x, m):
            if m.sum() == 0:
                return torch.zeros((), device=device)
            return x[:, m].mean().detach()

        max_r = torch.tensor(10.0, device=device)
        metrics = {
            "r_low": torch.clamp_max(band_mean(r, low), max_r),
            "r_mid": torch.clamp_max(band_mean(r, mid), max_r),
            "r_high": torch.clamp_max(band_mean(r, high), max_r),
            "lpe_low": band_mean(lpe.abs(), low),
            "lpe_mid": band_mean(lpe.abs(), mid),
            "lpe_high": band_mean(lpe.abs(), high),
        }
        return metrics

    def _bsp2d_per_time(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Compute BSP loss and diagnostics per time step.

        y_pred, y_true: [B, T, C, H, W]
        Returns:
          metrics = {"bsp": [T], ..., diagnostics}
        """
        B, T, C, H, W = y_pred.shape
        device = y_pred.device
        eps = self.eps

        # windowing
        yp = apply_hann_window(y_pred.to(torch.float32), H, W)
        yt = apply_hann_window(y_true.to(torch.float32), H, W)

        # FFT
        X = rfft2(yp, dim=(-2, -1), norm="ortho")  # [B, T, C, Hf, Wf]
        Y = rfft2(yt, dim=(-2, -1), norm="ortho")

        Hf, Wf = X.shape[-2], X.shape[-1]
        bin_index, mask, counts, n_bins = radial_bins_rfft(
            Hf, Wf, device=device, n_bins=self.n_bins
        )
        self.n_bins = n_bins
        edges = torch.linspace(1e-8, 1.0000001, n_bins + 1, device=device)
        self.k = 0.5 * (edges[:-1] + edges[1:])
        valid = self.k <= self.kmax_frac

        self.k = self.k[valid]

        # power spectra
        PX = X.real**2 + X.imag**2  # [B,T,C,Hf,Wf]
        PY = Y.real**2 + Y.imag**2

        flat_idx = bin_index[mask].flatten()  # [n_mask]
        mask_flat = mask.flatten()  # [Hf*Wf]
        counts = counts.to(device)  # [n_bins]

        def reduce_btc(Z):
            # Z: [B,T,C,Hf,Wf] -> [BTC, n_bins]
            Zbtc = Z.reshape(B * T * C, Hf * Wf)[:, mask_flat]  # [BTC, n_mask]
            sums = torch.zeros(
                B * T * C,
                self.n_bins,
                device=device,
                dtype=Z.dtype,
            )
            sums.index_add_(1, flat_idx, Zbtc)
            return sums / counts  # broadcast counts: [n_bins]

        E_pred = reduce_btc(PX).clamp_min(eps)  # [BTC, n_bins]
        E_true = reduce_btc(PY).clamp_min(eps)

        E_pred = E_pred[:, valid]
        E_true = E_true[:, valid]

        # BSP kernel: (1 - E_pred/E_true)^2
        ratio = (E_pred + eps) / (E_true + eps)
        bsp_bins = (1.0 - ratio) ** 2  # [BTC, n_bins]

        # optional bin weights λ_k ∝ k^gamma
        if self.gamma != 0.0:
            lam = self.k**self.gamma
            lam = lam / lam.mean().clamp_min(eps)
            bsp_bins = bsp_bins * lam  # broadcast over BTC

        # average over bins
        bsp_btc = bsp_bins.mean(dim=-1)  # [BTC]

        # reshape back to [B,T,C] if needed and then average over C
        bsp_bt = bsp_btc.view(B, T, C).mean(dim=2)  # [B,T]
        bsp_t = bsp_bt.mean(dim=0)  # [T]

        # diagnostics
        metrics = self._diag(E_pred, E_true, device)
        metrics["bsp"] = bsp_t  # per time step

        return metrics

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        y_true = y_true_full
        y_pred = y_pred_full

        # normalise to [B,T,C,H,W]
        if y_true.dim() == 4:
            # [B,C,H,W] -> [B,1,C,H,W]
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)
        elif y_true.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input, got {y_true.shape}")

        # pixel MSE per time step: [T]
        mse_t = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=[0, 2, 3, 4])

        # BSP term
        bsp_metrics = self._bsp2d_per_time(y_pred, y_true)  # contains "bsp": [T]
        bsp_t = bsp_metrics["bsp"]  # [T]
        bsp_loss = bsp_t.mean()

        combined_loss = mse_t.mean() + self.lambda_bsp * bsp_loss

        assert torch.isfinite(combined_loss), f"Non-finite loss: {combined_loss}"

        mse_loss = mse_t.mean()

        out = {
            "loss": combined_loss,
            "mse": mse_loss,
            "bsp_mse_ratio": bsp_loss / (mse_loss + 1e-12),
        }
        out.update(bsp_metrics)  # adds bsp, r_* and lpe_*

        return out
