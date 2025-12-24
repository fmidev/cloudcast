import math
import torch
import re
import torch.nn.functional as F
from torch import nn
from torch.fft import rfft2, rfftfreq, fftfreq
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import lightning as L
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
)


def _ensure_bchw(y):
    # Accept [B,C,H,W] or [C,H,W]. Return [B,C,H,W]
    if y.dim() == 4:
        return y
    elif y.dim() == 3:
        C, H, W = y.shape
        return y.view(1, C, H, W)

    raise ValueError(f"Expected 3D or 4D, got {y.shape}")


def _ensure_btchw(y):
    # Accept [B,T,C,H,W] or [B,C,H,W]. Return [B,T,C,H,W], T>=1
    if y.dim() == 5:
        return y
    elif y.dim() == 4:
        B, C, H, W = y.shape
        return y.view(B, 1, C, H, W)

    raise ValueError(f"Expected 4D or 5D, got {y.shape}")


def _mae(y, x):
    return (y - x).abs().mean().item()


def _bias(y, x):
    return (y - x).mean().item()


def _radial_bins(Hf, Wf, device, n_bins=None):
    # For rfft grid: spatial size in FFT domain is [Hf, Wf], where Wf = W//2 + 1
    fy = fftfreq(Hf, d=1.0, device=device)  # [-0.5,0.5)
    fx = rfftfreq(2 * (Wf - 1), d=1.0, device=device)  # [0,0.5]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    r = r / r.max().clamp(min=1e-8)  # normalize to [0,1]
    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)
    edges = torch.linspace(0, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index = bin_index.reshape(Hf, Wf).clamp(0, n_bins - 1)
    counts = torch.bincount(bin_index.flatten(), minlength=n_bins).clamp(min=1)
    return bin_index, counts, n_bins


@torch.no_grad
def _compute_conditional_bias(
    initial_cc: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor
):
    """
    Computes conditional bias per rollout step, binned by initial_cc.
    Returns keys like:
      conditional_bias_clear, conditional_bias_clear, ...
    """
    bins = [
        ("clear", 0.0, 0.125),
        ("scattered", 0.125, 0.5),
        ("broken", 0.5, 0.875),
        ("overcast", 0.875, 1.0),
    ]
    assert y_pred.shape == y_true.shape

    y_pred = _ensure_btchw(y_pred)  # [B,T,C,H,W]
    y_true = _ensure_btchw(y_true)  # [B,T,C,H,W]
    B, T, C, H, W = y_pred.shape

    conditioning = []
    for t in range(T):
        if t == 0:
            conditioning.append(initial_cc)  # [B,C,H,W]
        else:
            conditioning.append(y_true[:, t - 1])  # [B,C,H,W]

    conditioning = torch.stack(conditioning, dim=1)  # [B,T,C,H,W]

    results = {}

    for name, low, high in bins:
        if name == "overcast":
            mask = (conditioning >= low) & (conditioning <= high)
        else:
            mask = (conditioning >= low) & (conditioning < high)

        n_pixels = mask.sum()
        if n_pixels > 0:
            bias = (y_pred[mask] - y_true[mask]).mean()
            results[f"conditional_bias_{name}"] = float(bias.item())
        else:
            results[f"conditional_bias_{name}"] = float("nan")

    return results


@torch.no_grad()
def _ssim(y_pred: torch.Tensor, y_true: torch.Tensor):
    return structural_similarity_index_measure(
        _ensure_bchw(y_pred), _ensure_bchw(y_true)
    ).item()


@torch.no_grad()
def _psnr(y_pred: torch.Tensor, y_true: torch.Tensor):
    psnr = peak_signal_noise_ratio(y_pred, y_true, data_range=2.0)
    return (1 - torch.exp(-psnr / 20)).item()


@torch.no_grad()
def _compute_change_metrics(
    y_pred_btchw: torch.Tensor,
    y_true_btchw: torch.Tensor,
    prev_true_bchw: torch.Tensor | None = None,
    eps_change: float = 2e-3,
    eps_div: float = 1e-8,
):
    """
    Returns dict with averaged per-step scalars:
      - tendency_corr:  {f"t{t}": float}
      - stationarity_ratio: {f"t{t}": float}

    For t = 0 we use prev_true_bchw as the baseline if provided.
    For t >= 1 we use y_true[:, t-1] as the baseline.
    Works for any T >= 1. Expects shapes:
      y_*: [B, T, C, H, W], prev_true_bchw: [B, C, H, W] or None
    """
    y_pred = _ensure_btchw(y_pred_btchw)  # [B,T,C,H,W]
    y_true = _ensure_btchw(y_true_btchw)  # [B,T,C,H,W]
    B, T, C, H, W = y_pred.shape

    # Flatten utility
    def _flat(t):  # [B,C,H,W] -> [B, N]
        return t.reshape(B, -1)

    tend_corr = []
    stat_ratio = []

    for t in range(T):
        # Choose baseline
        if t == 0:
            if prev_true_bchw is None:
                # Cannot compute t0 without baseline; skip it
                continue
            base = prev_true_bchw  # [B,C,H,W]
        else:
            base = y_true[:, t - 1]  # [B,C,H,W]

        dy_p = y_pred[:, t] - base  # [B,C,H,W]
        dy_t = y_true[:, t] - base  # [B,C,H,W]

        # --- Tendency correlation (Pearson over pixels, then batch-mean) ---
        X = _flat(dy_p)  # [B,N]
        Y = _flat(dy_t)  # [B,N]
        Xc = X - X.mean(dim=1, keepdim=True)
        Yc = Y - Y.mean(dim=1, keepdim=True)
        num = (Xc * Yc).sum(dim=1)
        den = (
            Xc.square().sum(dim=1).clamp_min(eps_div).sqrt()
            * Yc.square().sum(dim=1).clamp_min(eps_div).sqrt()
        )
        corr_b = (num / den).clamp(-1.0, 1.0)  # [B]
        tend_corr.append(float(corr_b.mean().item()))

        # --- Stationarity ratio (#changed_pred / #changed_true) ---
        # fraction of pixels with |Δ| > eps_change, per-sample then ratio
        changed_p = (dy_p.abs() > eps_change).to(torch.float32)  # [B,C,H,W]
        changed_t = (dy_t.abs() > eps_change).to(torch.float32)
        frac_p = changed_p.mean(dim=(1, 2, 3))  # [B]
        frac_t = changed_t.mean(dim=(1, 2, 3)).clamp_min(eps_div)
        ratio_b = frac_p / frac_t  # [B]
        stat_ratio.append(float(ratio_b.mean().item()))

    return {
        "tendency_correlation": torch.tensor(tend_corr).mean(),
        "stationarity_ratio": torch.tensor(stat_ratio).mean(),
    }


@torch.no_grad()
def _psd_coherence_once(
    y_pred, y_true, top_frac=0.20, k_mid=0.30, k_high=0.60, eps=1e-8
):
    """
    Inputs: [C,H,W] tensors (single frame). Returns dict scalars.
    """
    device = y_pred.device
    dtype = y_pred.dtype
    C, H, W = y_pred.shape

    # FFT on each channel; use orthonormal to keep scales stable
    X = rfft2(y_pred, norm="ortho")  # [C,H,Wf]
    Y = rfft2(y_true, norm="ortho")  # [C,H,Wf]
    Hf, Wf = X.shape[-2], X.shape[-1]

    # PSD per-channel then average across channels
    PX = (X.real**2 + X.imag**2).mean(dim=0)  # [Hf,Wf]
    PY = (Y.real**2 + Y.imag**2).mean(dim=0)
    Cxy = (X * torch.conj(Y)).mean(dim=0).real  # [Hf,Wf], real cross-power

    # Radial binning
    bin_index, counts, n_bins = _radial_bins(Hf, Wf, device)
    flat_idx = bin_index.flatten()

    def _bin_mean(Z):
        Zb = Z.reshape(-1, Hf * Wf)
        sums = torch.zeros(Zb.shape[0], n_bins, dtype=Zb.dtype, device=device)
        sums.index_add_(1, flat_idx, Zb)
        return sums / counts  # [1,n_bins] in our usage

    PSDx = _bin_mean(PX[None, ...]).squeeze(0)  # [n_bins]
    PSDy = _bin_mean(PY[None, ...]).squeeze(0)
    Rxy = _bin_mean(Cxy[None, ...]).squeeze(0)

    # High-k power ratio (top_frac of bins)
    k0 = int((1.0 - top_frac) * n_bins)
    k0 = max(0, min(n_bins - 1, k0))
    hi_pred = PSDx[k0:].mean()
    hi_true = PSDy[k0:].mean()
    high_k_ratio = (hi_pred / (hi_true + eps)).item()

    # Coherence as Rxy / sqrt(Px*Py)
    denom = (PSDx.sqrt() * PSDy.sqrt()).clamp_min(eps)
    Coh = (Rxy / denom).clamp(-1.0, 1.0)  # [n_bins]

    def _pick_band(target):
        idx = int(target * (n_bins - 1))
        idx = max(0, min(n_bins - 1, idx))
        return Coh[idx].item()

    coh_mid = _pick_band(k_mid)
    coh_high = _pick_band(k_high)

    return {
        "high_k_power_ratio": high_k_ratio,
        "coherence_mid": coh_mid,
        "coherence_high": coh_high,
    }


@torch.no_grad()
def _fss_once(y_pred, y_true, category, radius, eps=1e-8, less_than=False):
    """
    Fraction Skill Score at one threshold & radius on a single frame.
    y_pred, y_true: [C,H,W] or [H,W] with values in [0,1] (cloud cover/occupancy).
    We binarize at threshold, then compute fractional cover via box filter (ones kernel).
    """
    if y_pred.dim() == 3:
        # average channels first (if multiple)
        y_pred = y_pred.mean(dim=0)
        y_true = y_true.mean(dim=0)
    # Binarize exceedance
    if category == "overcast":
        yp = (y_pred >= 0.875).to(torch.float32)[None, None]  # [1,1,H,W]
        yt = (y_true >= 0.875).to(torch.float32)[None, None]
    elif category == "broken":
        yp = ((y_pred < 0.875) & (y_pred >= 0.5)).to(torch.float32)[None, None]
        yt = ((y_true < 0.875) & (y_true >= 0.5)).to(torch.float32)[None, None]
    elif category == "scattered":
        yp = ((y_pred < 0.5) & (y_pred >= 0.125)).to(torch.float32)[None, None]
        yt = ((y_true < 0.5) & (y_true >= 0.125)).to(torch.float32)[None, None]
    else:
        assert category == "clear"
        yp = (y_pred < 0.125).to(torch.float32)[None, None]
        yt = (y_true < 0.125).to(torch.float32)[None, None]

    k = 2 * radius + 1
    kernel = torch.ones((1, 1, k, k), device=yp.device) / (k * k)
    pad = radius
    # fractional coverage in window
    yp_f = F.conv2d(F.pad(yp, (pad, pad, pad, pad), mode="reflect"), kernel)
    yt_f = F.conv2d(F.pad(yt, (pad, pad, pad, pad), mode="reflect"), kernel)

    num = (yp_f - yt_f).pow(2).mean()
    den = (yt_f.pow(2) + yp_f.pow(2)).mean() + eps
    fss = 1.0 - (num / den)
    return fss.item()


@torch.no_grad()
def _psd_anomaly_once(y_pred, y_true, lo_px=12.0, hi_px=6.0, eps=1e-8):
    """
    Mean PSD anomaly (log10(pred/obs)) in an annulus defined by pixel wavelengths
    [lo_px, hi_px]. With 5 km grid, 12 px ~ 60 km, 6 px ~ 30 km.
    Inputs: y_* [C,H,W] (single frame). Returns float.
    """
    device = y_pred.device
    y_pred = y_pred.to(torch.float32)
    y_true = y_true.to(torch.float32)

    X = rfft2(y_pred, norm="ortho")  # [C,H,Wf]
    Y = rfft2(y_true, norm="ortho")
    # channel-mean PSD
    PSDx = (X.real**2 + X.imag**2).mean(dim=0)  # [Hf,Wf]
    PSDy = (Y.real**2 + Y.imag**2).mean(dim=0)

    Hf, Wf = PSDx.shape[-2], PSDx.shape[-1]
    bin_index, counts, n_bins = _radial_bins(Hf, Wf, device)
    flat_idx = bin_index.flatten()

    def _bin_mean(Z):
        Zb = Z.reshape(1, Hf * Wf)
        sums = torch.zeros(1, n_bins, dtype=Z.dtype, device=device)
        sums.index_add_(1, flat_idx, Zb)
        return (sums / counts).squeeze(0)  # [n_bins]

    bx = _bin_mean(PSDx).clamp_min(eps)
    by = _bin_mean(PSDy).clamp_min(eps)

    r = torch.linspace(0, 1, steps=n_bins, device=device, dtype=bx.dtype)
    inv_sqrt2 = 1.0 / math.sqrt(0.5**2 + 0.5**2)
    rn_lo = (1.0 / lo_px) * inv_sqrt2
    rn_hi = (1.0 / hi_px) * inv_sqrt2
    # hard mask (simple and robust)
    mask = (r >= rn_lo) & (r <= rn_hi)
    if not mask.any():
        return 0.0

    anom = torch.log10(bx[mask] / by[mask])
    return float(anom.mean().item())


@torch.no_grad()
def _compute_metrics_pack(x_hist_btchw, y_pred_btchw, y_true_btchw, clip_high_k=8.0):
    """
    Rollout-aware metrics pack.

    Inputs:
      x_hist: [B,Th,C,H,W] or [B,C,H,W]
      y_pred: [B,T,C,H,W] or [B,C,H,W]
      y_true: [B,T,C,H,W] or [B,C,H,W]

    Outputs:
      - Per-step metrics keyed as "<metric>_r{t}" for t=0..T-1
      - Overall (time-mean) metrics keyed as "<metric>" (mean over t)
      - Change metrics already include per-step keys from _compute_change_metrics_per_step
      - Conditional bias per-step keys from _compute_conditional_bias
    """
    x_hist = _ensure_btchw(x_hist_btchw)
    y_pred = _ensure_btchw(y_pred_btchw)
    y_true = _ensure_btchw(y_true_btchw)

    B, T, C, H, W = y_pred.shape

    per_step = {t: {} for t in range(T)}

    for t in range(T):
        yp = y_pred[:, t]  # [B,C,H,W]
        yt = y_true[:, t]

        (
            bias_list,
            mae_list,
            hk_list,
            coh_mid_list,
            coh_hi_list,
            fss_ovc_list_30,
            fss_bkn_list_30,
            fss_sct_list_30,
            fss_cavok_list_30,
            psd_anom_list_60_100,
            psd_anom_list_30_60,
            psd_anom_list_10_30,
            ssim_list,
            psnr_list,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])

        for b in range(B):
            yp_b = yp[b]
            yt_b = yt[b]

            bias_list.append(_bias(yp_b, yt_b))
            mae_list.append(_mae(yp_b, yt_b))

            coh = _psd_coherence_once(yp_b, yt_b)
            coh_mid_list.append(coh["coherence_mid"])
            coh_hi_list.append(coh["coherence_high"])
            hk_list.append(coh["high_k_power_ratio"])

            fss_ovc_list_30.append(_fss_once(yp_b, yt_b, "overcast", radius=6))
            fss_bkn_list_30.append(_fss_once(yp_b, yt_b, "broken", radius=6))
            fss_sct_list_30.append(_fss_once(yp_b, yt_b, "scattered", radius=6))
            fss_cavok_list_30.append(_fss_once(yp_b, yt_b, "clear", radius=6))

            psd_anom_list_60_100.append(
                _psd_anomaly_once(yp_b, yt_b, lo_px=20.0, hi_px=12.0)
            )
            psd_anom_list_30_60.append(
                _psd_anomaly_once(yp_b, yt_b, lo_px=12.0, hi_px=6.0)
            )
            psd_anom_list_10_30.append(
                _psd_anomaly_once(yp_b, yt_b, lo_px=6.0, hi_px=2.0)
            )

            ssim_list.append(_ssim(yp_b, yt_b))
            psnr_list.append(_psnr(yp_b, yt_b))

        per_step[t] = {
            "mae": float(sum(mae_list) / len(mae_list)),
            "bias": float(sum(bias_list) / len(bias_list)),
            "high_k_power_ratio": float(min(sum(hk_list) / len(hk_list), clip_high_k)),
            "coherence_mid": float(sum(coh_mid_list) / len(coh_mid_list)),
            "coherence_high": float(sum(coh_hi_list) / len(coh_hi_list)),
            "fss_overcast_30km": float(sum(fss_ovc_list_30) / len(fss_ovc_list_30)),
            "fss_broken_30km": float(sum(fss_bkn_list_30) / len(fss_bkn_list_30)),
            "fss_scattered_30km": float(sum(fss_sct_list_30) / len(fss_sct_list_30)),
            "fss_clear_30km": float(sum(fss_cavok_list_30) / len(fss_cavok_list_30)),
            "psd_anom_60_100km": float(
                sum(psd_anom_list_60_100) / len(psd_anom_list_60_100)
            ),
            "psd_anom_30_60km": float(
                sum(psd_anom_list_30_60) / len(psd_anom_list_30_60)
            ),
            "psd_anom_10_30km": float(
                sum(psd_anom_list_10_30) / len(psd_anom_list_10_30)
            ),
            "ssim": float(sum(ssim_list) / len(ssim_list)),
            "psnr": float(sum(psnr_list) / len(psnr_list)),
        }

    out = {}
    for t in range(T):
        for k, v in per_step[t].items():
            out[f"{k}_r{t}"] = v

    for k in per_step[0].keys():
        out[k] = float(sum(per_step[t][k] for t in range(T)) / T)

    prev_true = x_hist[:, -1]  # [B,C,H,W]
    cm = _compute_change_metrics(
        y_pred, y_true, prev_true_bchw=prev_true, eps_change=1e-3
    )
    out.update(cm)

    out.update(_compute_conditional_bias(prev_true, y_pred, y_true))

    return out


class EarlyWarningMetricsCallback(L.Callback):
    """
    Logs quick diagnostics to MLflow at epoch end using predictions already
    stored on the pl_module:
      - latest_train_predictions, latest_train_data = (x, y)
      - latest_val_predictions,   latest_val_data   = (x, y)

    It does NOT run any forward passes itself.
    """

    def __init__(
        self,
        fss_threshold: float = 0.5,
        fss_radius: int = 10,
        log_train: bool = True,
        log_val: bool = True,
    ):
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val

        self._rollout_key = re.compile(r"_r\d+$")

    def _split_mean_vs_rollout(self, metrics: dict) -> tuple[dict, dict]:
        mean_metrics = {}
        rollout_metrics = {}
        for k, v in metrics.items():
            if self._rollout_key.search(k):
                rollout_metrics[k] = v
            else:
                mean_metrics[k] = v
        return mean_metrics, rollout_metrics

    def _count_rollout_steps(self, rollout_metrics: dict) -> int:
        """Count the number of unique rollout steps from metric keys."""
        if not rollout_metrics:
            return 0

        rollout_indices = set()
        for key in rollout_metrics.keys():
            match = self._rollout_key.search(key)
            if match:
                # Extract the number from '_rN'
                rollout_idx = int(match.group()[2:])  # Remove '_r' prefix
                rollout_indices.add(rollout_idx)

        return len(rollout_indices)

    def _mlflow_log_metrics(self, pl_module, metrics: dict, stage: str, root: str):
        for k, v in metrics.items():
            pl_module.log(
                f"{root}/{stage}/{k}",
                float(v),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if (
            self.log_train
            and pl_module.latest_train_predictions is not None
            and pl_module.latest_train_data is not None
            and pl_module.latest_train_data
        ):
            with torch.no_grad():
                y_pred = pl_module.latest_train_predictions
                x_hist, y_true = pl_module.latest_train_data
                # Ensure on same device & dtype if needed
                metrics = _compute_metrics_pack(
                    x_hist,
                    y_pred,
                    y_true,
                )

            mean_metrics, rollout_metrics = self._split_mean_vs_rollout(metrics)
            self._mlflow_log_metrics(
                pl_module, mean_metrics, stage="train", root="metrics"
            )
            if self._count_rollout_steps(rollout_metrics) > 1:
                self._mlflow_log_metrics(
                    pl_module, rollout_metrics, stage="train", root="metrics_rollout"
                )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if (
            self.log_val
            and pl_module.latest_val_predictions is not None
            and pl_module.latest_val_data is not None
        ):
            with torch.no_grad():
                y_pred = pl_module.latest_val_predictions
                x_hist, y_true = pl_module.latest_val_data
                metrics = _compute_metrics_pack(
                    x_hist,
                    y_pred,
                    y_true,
                )

            mean_metrics, rollout_metrics = self._split_mean_vs_rollout(metrics)
            self._mlflow_log_metrics(
                pl_module, mean_metrics, stage="val", root="metrics"
            )
            if self._count_rollout_steps(rollout_metrics) > 1:
                self._mlflow_log_metrics(
                    pl_module, rollout_metrics, stage="val", root="metrics_rollout"
                )
