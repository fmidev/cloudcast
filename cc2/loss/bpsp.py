import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2

from swinu.util import radial_bins_rfft, apply_hann_window


class BPSPLoss(nn.Module):
    """
    Binned Polar Spectral Power (BPSP) loss
    Adds *direction sectors* (theta bins) on top of radial bins

    Notes:
      - Direction binning is NOT phase/coherence: we only use |F|^2.
      - n_theta=1 reproduces radial-only behaviour (but still ratio-based).
    """

    def __init__(
        self,
        n_bins: int | None = None,  # radial bins
        n_theta: int = 8,  # direction sectors; 1 => no directional split
        beta: float = 1.0,  # k^beta weighting
        kmax_frac: float = 0.707,  # drop bins beyond this normalized k
    ):
        super().__init__()
        assert n_theta >= 1
        assert 0.0 < kmax_frac <= 1.0

        self.eps = 1e-8
        self.n_bins = n_bins
        self.n_theta = int(n_theta)
        self.beta = float(beta)
        self.kmax_frac = float(kmax_frac)

        # cached per-(H,W,device) binning
        self._cache_key = None
        self._bin_index = None  # [Hf,Wf] long, combined polar bin id
        self._mask = None  # [Hf,Wf] bool, valid freq pixels for reduction
        self._counts = None  # [n_bins_total] float, counts per polar bin
        self._flat_idx = None  # [N_masked] long, flattened bin ids under mask
        self.k = None  # [n_bins_valid] float, normalized radial k repeated across theta

    def _build_polar_bins(self, H: int, W: int, device: torch.device):
        Hf = H
        Wf = W // 2 + 1

        # 1) radial bins from existing helper (works on rfft grid)
        bin_index_r, mask, counts_r, n_rbins = radial_bins_rfft(
            Hf, Wf, device, self.n_bins
        )

        # 2) direction bins on the rfft grid
        # rfft2 output has kx >= 0 only; ky has both signs (fftfreq).
        # Reconstruct the full spatial width from Wf = W//2 + 1:
        W_full = (Wf - 1) * 2

        ky = torch.fft.fftfreq(Hf, d=1.0, device=device) * Hf
        kx = torch.fft.rfftfreq(W, d=1.0, device=device) * W

        KY, KX = torch.meshgrid(ky, kx, indexing="ij")  # [Hf,Wf]

        theta = torch.atan2(KY, KX)  # ~[-pi/2, pi/2] on rFFT half-plane
        theta01 = (theta + math.pi / 2) / math.pi  # map to [0,1]
        theta_bin = torch.clamp(
            (theta01 * self.n_theta).floor().long(), 0, self.n_theta - 1
        )

        # 3) combine radial + theta into a single bin id
        n_bins_total = n_rbins * self.n_theta
        bin_index = bin_index_r * self.n_theta + theta_bin  # [Hf,Wf] long

        # counts per polar bin (some polar bins can be empty, clamp later)
        flat_idx = bin_index[mask].flatten()
        counts = torch.bincount(flat_idx, minlength=n_bins_total).to(
            device=device, dtype=torch.float32
        )
        counts = counts.clamp_min(1.0)

        # 4) build normalized k vector and "valid bins" mask
        k_r = torch.linspace(0.0, 1.0, n_rbins, device=device)
        k_all = k_r.repeat_interleave(self.n_theta)  # length n_bins_total
        valid_bins = k_all <= self.kmax_frac

        # cache everything
        self._bin_index = bin_index
        self._mask = mask
        self._counts = counts
        self._flat_idx = flat_idx
        self.k = k_all[valid_bins]

        self._n_rbins = n_rbins
        self._n_bins_total = n_bins_total
        self._valid_bins = valid_bins

    def _build_cache(self, H: int, W: int, device: torch.device):
        # rfft2 output shape on last dim
        Hf = H
        Wf = W // 2 + 1
        key = (
            Hf,
            Wf,
            W,
            device.type,
            device.index,
            self.n_bins,
            self.n_theta,
            self.kmax_frac,
        )

        if self._cache_key != key:
            self._build_polar_bins(H, W, device)
            self._cache_key = key

    def _diag(
        self, PSDx: torch.Tensor, PSDy: torch.Tensor, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        # self.k is [n_bins_valid]
        k = self.k.to(device=device, dtype=PSDx.dtype)

        low = k < 0.20
        mid = (k >= 0.20) & (k < 0.60)
        high = k >= 0.60

        ratio = (PSDx + self.eps) / (PSDy + self.eps)
        lpe = torch.log(PSDx + self.eps) - torch.log(PSDy + self.eps)

        def band_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            if m.sum().item() == 0:
                return torch.zeros((), device=x.device, dtype=x.dtype)
            return x[:, m].mean().detach()

        return {
            "r_low": band_mean(ratio, low),
            "r_mid": band_mean(ratio, mid),
            "r_high": band_mean(ratio, high),
            "lpe_low": band_mean(lpe.abs(), low),
            "lpe_mid": band_mean(lpe.abs(), mid),
            "lpe_high": band_mean(lpe.abs(), high),
        }

    def _spec2d_per_time(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        y_pred, y_true: [B,T,C,H,W], C must be 1
        returns dict including 'bpsp': [T] plus diag scalars.
        """
        B, T, C, H, W = y_pred.shape
        assert C == 1, f"Support only one output channel, got C={C}"

        device = y_pred.device
        self._build_cache(H, W, device)

        yp = apply_hann_window(y_pred, H, W).to(torch.float32)
        yt = apply_hann_window(y_true, H, W).to(torch.float32)

        X = rfft2(yp, dim=(-2, -1), norm="ortho").squeeze(dim=2)  # [B,T,Hf,Wf]
        Y = rfft2(yt, dim=(-2, -1), norm="ortho").squeeze(dim=2)

        PX = X.real**2 + X.imag**2  # power
        PY = Y.real**2 + Y.imag**2

        # One-sided PSD correction for rfft2 along x
        if W % 2 == 0:
            PX[..., 1:-1] *= 2.0
            PY[..., 1:-1] *= 2.0
        else:
            PX[..., 1:] *= 2.0
            PY[..., 1:] *= 2.0

        Hf, Wf = PX.shape[-2], PX.shape[-1]
        mask = self._mask
        flat_idx = self._flat_idx
        counts = self._counts.to(device=PX.device, dtype=PX.dtype)

        n_bins_total = self._n_bins_total
        valid_bins = self._valid_bins

        def reduce_bt(Z: torch.Tensor) -> torch.Tensor:
            # Z: [B,T,Hf,Wf] -> [B*T, N_masked]
            Zbt = Z.reshape(B * T, Hf * Wf)[:, mask.flatten()]
            sums = torch.zeros((B * T, n_bins_total), device=Z.device, dtype=Z.dtype)
            sums.index_add_(1, flat_idx, Zbt)
            return sums / counts

        PSDx = reduce_bt(PX).clamp_min(self.eps)[:, valid_bins]  # [B*T, n_bins_valid]
        PSDy = reduce_bt(PY).clamp_min(self.eps)[:, valid_bins]

        # --- BSP-style ratio penalty on binned power ---
        bpsp_bin = (torch.log1p(PSDx / self.eps) - torch.log1p(PSDy / self.eps)) ** 2

        k = self.k.to(device=bpsp_bin.device, dtype=bpsp_bin.dtype)  # [n_bins_valid]
        if self.beta != 0.0:
            w = k**self.beta
            w = w / (w.mean() + 1e-12)
            bpsp_bin = bpsp_bin * w

        bpsp_bt = bpsp_bin.mean(dim=1)  # [B*T]
        bpsp_t = bpsp_bt.view(B, T).mean(dim=0)  # [T]

        metrics = self._diag(PSDx, PSDy, device)
        metrics["bpsp"] = bpsp_t
        return metrics

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        """
        Accepts:
          - [B,C,H,W]  (single step)  OR
          - [B,T,C,H,W]
        Returns dict with scalar losses + diagnostics.
        """
        y_true = y_true_full
        y_pred = y_pred_full

        # Normalize shapes to [B,T,C,H,W]
        if y_true.dim() == 4:
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)
        assert (
            y_true.dim() == 5 and y_pred.dim() == 5
        ), "Expected [B,T,C,H,W] or [B,C,H,W]"
        assert y_true.shape == y_pred.shape

        # spectral metrics (includes bpsp_t: [T])
        spec_metrics = self._spec2d_per_time(y_pred, y_true)
        bpsp_t = spec_metrics["bpsp"]

        # scalar summaries
        loss = {
            "loss": bpsp_t.mean(),
        }
        loss.update(spec_metrics)

        assert torch.isfinite(loss["loss"]), f"Non-finite loss: {loss['loss']}"
        return loss
