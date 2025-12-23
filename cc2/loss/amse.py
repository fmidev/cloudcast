import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2
from swinu.util import radial_bins_rfft


class AMSELoss(nn.Module):
    def __init__(
        self,
        n_bins: int | None = None,
    ):
        super().__init__()
        self.n_bins = n_bins

    def _diag(self, amp, coh, PSDx, PSDy, device):

        k = torch.linspace(0, 1, self.n_bins, device=device)
        low = k < 0.20
        mid = (k >= 0.20) & (k < 0.45)
        high = k >= 0.45

        r = PSDx / (PSDy + 1e-12)
        lpe = torch.log(PSDx) - torch.log(PSDy)

        def band_mean(x, m):
            return x[:, m].mean().detach()

        max_r = torch.tensor(10).to(device)
        metrics = {
            "r_low": torch.clamp_max(max_r, band_mean(r, low)),
            "r_mid": torch.clamp_max(max_r, band_mean(r, mid)),
            "r_high": torch.clamp_max(max_r, band_mean(r, high)),
            "coh_low": band_mean(coh, low),
            "coh_mid": band_mean(coh, mid),
            "coh_high": band_mean(coh, high),
            "lpe_low": band_mean(lpe.abs(), low),
            "lpe_mid": band_mean(lpe.abs(), mid),
            "lpe_high": band_mean(lpe.abs(), high),
            "coh_high_std": coh[:, high].std().detach(),
            "lpe_high_std": lpe[:, high].std().detach(),
        }

        return metrics

    def _one_sided_weights(self, Wf: int, device: str):
        # double interior freq columns (DC & Nyquist stay 1.0)
        w = torch.ones(Wf, device=device)
        if Wf > 1:
            w[1 : Wf - 1] = 2.0
        return w

    def _parseval_energy_diff(self, X, Y):
        # X,Y: complex [B,T,Hf,Wf]; energy of (X-Y) with one-sided weight
        Wf = X.shape[-1]
        w = self._one_sided_weights(Wf, X.device)
        D = X - Y
        E = (D.real.square() + D.imag.square()) * w  # [B,T,Hf,Wf]
        return E.sum(dim=(-2, -1))  # sum over Hf,Wf

    def _apply_window(self, field: torch.tensor, H: int, W: int):
        wh = torch.hann_window(H, device=field.device).unsqueeze(1)  # (H, 1)
        ww = torch.hann_window(W, device=field.device).unsqueeze(0)  # (1, W)
        win = (wh @ ww).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        win_rms = (win**2).mean().sqrt()
        return field * win / win_rms

    def _amse2d_per_time(self, y_pred: torch.tensor, y_true: torch.tensor):
        eps = 1e-8

        B, T, C, H, W = y_pred.shape

        assert C == 1, f"Support only one output channel (tcc), got: {C}"
        device = y_pred.device

        yp = self._apply_window(y_pred, H, W)
        yt = self._apply_window(y_true, H, W)

        yp = yp.to(torch.float32)
        yt = yt.to(torch.float32)

        # rfft2 over H,W; average PSD/cross over channels
        X = rfft2(yp, dim=(-2, -1), norm="ortho").squeeze(dim=2)  # [B,T,Hf,Wf]
        Y = rfft2(yt, dim=(-2, -1), norm="ortho").squeeze(dim=2)

        Es = (yp - yt).square().sum(dim=(-2, -1))  # [B,T,1]
        # spectral energy of diff (rfft2 one-sided)
        Ed = self._parseval_energy_diff(X.squeeze(2), Y.squeeze(2))  # [B,T]
        parseval_check = (Ed / (Es.squeeze(2) + 1e-12)).mean().detach()  # ~1.0

        Hf, Wf = X.shape[-2], X.shape[-1]
        bin_index, mask, counts, n_bins = radial_bins_rfft(Hf, Wf, device, self.n_bins)
        self.n_bins = n_bins

        # Magnitudes & cross, channel-mean
        PX = X.real**2 + X.imag**2  # [B,T,Hf,Wf]
        PY = Y.real**2 + Y.imag**2

        # Cross spectrum
        Sxy = X * torch.conj(Y)  # complex [B,T,Hf,Wf]

        flat_idx = bin_index[mask].flatten()

        def reduce_bt(Z):  # Z: [B,T,Hf,Wf] -> [B*T, n_bins]
            Zbt = Z.reshape(B * T, Hf * Wf)[:, mask.flatten()]
            sums = torch.zeros(B * T, self.n_bins, device=device, dtype=Z.dtype)
            sums.index_add_(1, flat_idx, Zbt)
            return sums / counts  # #fix3 but reverted

        PSDx = reduce_bt(PX).clamp_min(eps)  # [B*T, n_bins]
        PSDy = reduce_bt(PY).clamp_min(eps)
        Sxy_m = reduce_bt(Sxy)

        sqrtx = PSDx.sqrt()
        sqrty = PSDy.sqrt()
        denom = (sqrtx * sqrty).clamp_min(eps)

        # signed coherence (phase-aware)
        CohR = (Sxy_m.real / denom).clamp(-1, 1)

        amp_term = (sqrtx - sqrty) ** 2

        # fix #1: signed coherence (phase-aware)
        # fix #3: multiply still by 2.0, remove clamp
        coh_term = 2.0 * (1.0 - CohR) * torch.maximum(PSDx, PSDy)

        metrics = self._diag(amp_term, coh_term, PSDx, PSDy, device)

        amp_term = amp_term.mean(dim=1)
        coh_term = coh_term.mean(dim=1)

        amse_bt = amp_term + coh_term  # [B*T]

        amp_coh_ratio = amp_term / (coh_term + 1e-12)

        loss = {
            "amse": amse_bt.mean(),
            "coh": coh_term.mean(),
            "amp": amp_term.mean(),
            "amp_coh_ratio": amp_coh_ratio.mean(),
            "cohr_mean": CohR.mean().detach(),
            "mse_parseval_ratio": parseval_check,
        }

        loss.update(metrics)

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
