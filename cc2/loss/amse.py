import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, rfftfreq, fftfreq
from swinu.util import apply_hann_window


class AMSELoss(nn.Module):
    def __init__(
        self,
        n_bins: int | None = None,
    ):
        super().__init__()
        self.n_bins = n_bins

    def _one_sided_weights(self, Wf: int, device: str):
        # double interior freq columns (DC & Nyquist stay 1.0)
        w = torch.ones(Wf, device=device)
        if Wf > 1:
            w[1 : Wf - 1] = 2.0
        return w

    def _radial_bins_rfft(self, H: int, W: int, device, n_bins: int | None):
        # Frequency indices in "FFT bins" (not physical units). With dx=dy this is sufficient.
        ky = fftfreq(H, d=1.0, device=device) * H  # [-H/2..H/2)
        kx = rfftfreq(W, d=1.0, device=device) * W  # [0..W/2]

        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(KY**2 + KX**2)

        # Integer shell index: 0,1,2,... in grid units
        shell = torch.round(r).to(torch.int64)  # or torch.floor(r + 0.5)

        mask = shell >= 0
        max_shell = int(shell[mask].max().item()) if mask.any() else 0

        # Choose number of bins (shells). Reasonable default is up to Nyquist radius.
        if n_bins is None:
            n_bins = max_shell + 1
        else:
            # clamp shells above n_bins-1 into last bin to avoid dropping energy
            shell = torch.clamp(shell, max=n_bins - 1)

        counts = torch.bincount(shell[mask].flatten(), minlength=n_bins).clamp(min=1)
        return shell, mask, counts, n_bins

    def _amse2d_per_time(self, y_pred: torch.tensor, y_true: torch.tensor):
        eps = 1e-8

        B, T, C, H, W = y_pred.shape

        assert C == 1, f"Support only one output channel (tcc), got: {C}"
        device = y_pred.device

        yp = apply_hann_window(y_pred.to(torch.float32), H, W).squeeze(dim=2)
        yt = apply_hann_window(y_true.to(torch.float32), H, W).squeeze(dim=2)

        # rfft2 over H,W
        X = rfft2(yp, dim=(-2, -1), norm="ortho")  # [B,T,Hf,Wf]
        Y = rfft2(yt, dim=(-2, -1), norm="ortho")
        Hf, Wf = X.shape[-2], X.shape[-1]

        # one-sided weights along rFFT last dimension (kx>=0)
        w1 = self._one_sided_weights(Wf, X.device).view(1, 1, 1, Wf)

        PXw = (X.real.square() + X.imag.square()) * w1
        PYw = (Y.real.square() + Y.imag.square()) * w1
        PXYw = (X * torch.conj(Y)).real * w1

        shell, mask, counts, n_bins = self._radial_bins_rfft(
            H, W, device=device, n_bins=self.n_bins
        )
        self.n_bins = n_bins

        # shell/mask live on the rFFT grid [Hf,Wf] if implemented as shown previously
        assert shell.shape == (
            Hf,
            Wf,
        ), f"shell shape {shell.shape} must match rFFT grid {(Hf, Wf)}"
        assert mask.shape == (Hf, Wf)

        flat_idx = shell[mask].flatten().to(torch.int64)  # [Nmasked]
        mflat = mask.flatten()

        def bin_sum(Z: torch.Tensor) -> torch.Tensor:
            """Z: [B,T,Hf,Wf] -> [B*T,n_bins] (sum over all modes in each shell)"""
            Zbt = Z.reshape(B * T, Hf * Wf)[:, mflat]  # [B*T, Nmasked]
            out = torch.zeros(B * T, n_bins, device=device, dtype=Z.dtype)
            out.index_add_(1, flat_idx, Zbt)
            return out

        PSDx = bin_sum(PXw).clamp_min(eps)  # [B*T,n_bins] sums
        PSDy = bin_sum(PYw).clamp_min(eps)
        num = bin_sum(PXYw)  # [B*T,n_bins] sums

        # signed coherence per shell (paper definition uses Re of cross-sum / sqrt(PSD products))
        denom = torch.sqrt(PSDx * PSDy).clamp_min(eps)
        Coh = (num / denom).clamp(-1.0, 1.0)  # [B*T,n_bins]

        # AMSE terms per shell (paper eq. 6)
        amp_k = (PSDx.sqrt() - PSDy.sqrt()).square()  # [B*T,n_bins]
        coh_k = 2.0 * (1.0 - Coh) * torch.maximum(PSDx, PSDy)  # [B*T,n_bins]

        # sum over shells (paper uses Σ_k); optional normalization to per-pixel scale
        amse_bt = (amp_k + coh_k).sum(dim=1) / (H * W)  # [B*T]

        # aggregate over B*T
        amp_bt = amp_k.sum(dim=1) / (H * W)
        coh_bt = coh_k.sum(dim=1) / (H * W)

        amp_coh_ratio = amp_bt / (coh_bt + 1e-12)

        loss = {
            "amse": amse_bt.mean(),
            "coh": coh_bt.mean(),
            "amp": amp_bt.mean(),
            "amp_coh_ratio": amp_coh_ratio.mean(),
        }

        return loss

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        y_true = y_true_full
        y_pred = y_pred_full

        if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        loss = self._amse2d_per_time(y_pred, y_true)  # [T]

        assert torch.isfinite(loss["amse"]), f"Non-finite loss: {loss['amse']}"

        loss["loss"] = loss["amse"]

        return loss
