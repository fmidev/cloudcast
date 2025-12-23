import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2
from swinu.util import radial_bins_rfft, apply_hann_window


class DSELoss(nn.Module):
    """
    Combines pixel-wise Mean Squared Error (MSE) with the Direct Spectral Error (DSE).
    L_Total = L_MSE + lambda_dse * L_DSE
    """

    def __init__(
        self,
        lambda_dse: float = 1.0,  # Weight for the DSE component
        n_bins: int | None = None,
        beta: float = 1.0,  # Control wavenumber weighting
        kmax_frac: float = 0.707,  # remove all bins below nyquist (assuming 5km spacing)
    ):
        super().__init__()
        self.n_bins = n_bins
        self.lambda_dse = lambda_dse
        self.beta = beta
        self.k = None
        self.kmax_frac = kmax_frac

    def _diag(self, PSDx, PSDy, device):

        low = self.k < 0.20
        mid = (self.k >= 0.20) & (self.k < 0.45)
        high = self.k >= 0.45

        r = PSDx / (PSDy + 1e-12)
        lpe = torch.log(PSDx) - torch.log(PSDy)

        def band_mean(x, m):
            return x[:, m].mean().detach()

        max_r = torch.tensor(10).to(device)
        metrics = {
            "r_low": torch.clamp_max(max_r, band_mean(r, low)),
            "r_mid": torch.clamp_max(max_r, band_mean(r, mid)),
            "r_high": torch.clamp_max(max_r, band_mean(r, high)),
            "lpe_low": band_mean(lpe.abs(), low),
            "lpe_mid": band_mean(lpe.abs(), mid),
            "lpe_high": band_mean(lpe.abs(), high),
        }

        return metrics

    def _dse2d_per_time(self, y_pred: torch.tensor, y_true: torch.tensor):
        eps = 1e-8
        B, T, C, H, W = y_pred.shape
        assert C == 1, f"Support only one output channel (tcc), got: {C}"
        device = y_pred.device

        yp = apply_hann_window(y_pred, H, W)
        yt = apply_hann_window(y_true, H, W)

        yp = yp.to(torch.float32)
        yt = yt.to(torch.float32)

        # FFT and Binning setup
        X = rfft2(yp, dim=(-2, -1), norm="ortho").squeeze(dim=2)
        Y = rfft2(yt, dim=(-2, -1), norm="ortho").squeeze(dim=2)
        Hf, Wf = X.shape[-2], X.shape[-1]
        bin_index, mask, counts, n_bins = radial_bins_rfft(Hf, Wf, device, self.n_bins)
        self.n_bins = n_bins
        self.k = torch.linspace(0, 1, self.n_bins, device=device)
        valid = self.k <= self.kmax_frac
        self.k = self.k[valid]

        PX = X.real**2 + X.imag**2
        PY = Y.real**2 + Y.imag**2

        flat_idx = bin_index[mask].flatten()

        def reduce_bt(Z):
            Zbt = Z.reshape(B * T, Hf * Wf)[:, mask.flatten()]
            sums = torch.zeros(B * T, self.n_bins, device=device, dtype=Z.dtype)
            sums.index_add_(1, flat_idx, Zbt)
            return sums / counts

        PSDx = reduce_bt(PX).clamp_min(eps)
        PSDy = reduce_bt(PY).clamp_min(eps)

        PSDx = PSDx[:, valid]
        PSDy = PSDy[:, valid]

        # DSE Calculation
        sqrtx = PSDx.sqrt()
        sqrty = PSDy.sqrt()
        dse_bin = (sqrtx - sqrty) ** 2

        # Apply power law and normalize the weights
        w = (self.k**self.beta) / ((self.k**self.beta).mean())
        dse_bin = w * dse_bin

        dse_bt = dse_bin.mean(dim=1)
        dse_t = dse_bt.view(B, T).mean(dim=0)

        train_metrics = self._diag(PSDx, PSDy, device)

        train_metrics["dse"] = dse_t

        return train_metrics

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        y_true = y_true_full
        y_pred = y_pred_full

        if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        # 1. Calculate Pixel-wise MSE Loss
        # We average over Batch, Channel, Height, and Width, leaving the Time dimension [T]
        mse_loss_t = F.mse_loss(y_pred, y_true, reduction="none").mean(
            dim=[0, 2, 3, 4]
        )  # [T]

        # 2. Calculate Direct Spectral Error (DSE) Loss
        dse_metrics = self._dse2d_per_time(y_pred, y_true)  # [T]
        dse_loss = dse_metrics["dse"].mean()

        # 3. Combine Losses
        combined_loss = mse_loss_t.mean() + self.lambda_dse * dse_loss
        # combined_loss = combined_loss_t.mean()  # Final scalar loss (mean over Time)

        assert torch.isfinite(combined_loss), f"Non-finite loss: {combined_loss}"

        mse_loss = mse_loss_t.mean()

        loss = {
            "loss": combined_loss,
            "mse": mse_loss,
            "dse_mse_ratio": dse_loss / (mse_loss + 1e-12),
        }

        loss.update(dse_metrics)

        return loss
