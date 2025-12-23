import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2
from swinu.util import radial_bins_rfft, apply_hann_window


class DSELoss(nn.Module):
    def __init__(
        self,
        n_bins: int | None = None,
        beta: float = 1.0,  # Control wavenumber weighting
        kmax_frac: float = 0.707,  # remove all bins below nyquist (assuming 5km spacing)
    ):
        super().__init__()
        self.n_bins = n_bins
        self.beta = beta
        self.k = None
        self.kmax_frac = kmax_frac

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

        return dse_t

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        y_true = y_true_full
        y_pred = y_pred_full

        if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        dse_t = self._dse2d_per_time(y_pred, y_true)  # [T]
        dse_loss = dse_t.mean()

        assert torch.isfinite(dse_loss), f"Non-finite loss: {dse_loss}"

        loss = {
            "loss": dse_loss,
        }

        return loss
