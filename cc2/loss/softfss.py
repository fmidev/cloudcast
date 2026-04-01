import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftFSSLoss(nn.Module):
    """
    Differentiable multiscale FSS-style loss using soft cumulative masks.

    Expected input:
        y_pred_full, y_true_full in full-state probability space, shape
        [B, T, C, H, W] or [B, C, H, W], with C == 1.

    For thresholds t1 < t2 < t3:
        m_i(pred) = sigmoid((pred - t_i) / tau)
        m_i(true) = 1[true >= t_i]            # first implementation

    For each threshold and neighborhood size:
        pooled = avg_pool2d(mask, kernel_size=w, stride=1)

    Loss per threshold/scale:
        1 - FSS = sum((pooled_pred - pooled_true)^2) /
                  (sum(pooled_pred^2) + sum(pooled_true^2) + eps)
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.0625, 0.5625, 0.9375),
        window_sizes: tuple[int, ...] = (6, 12, 24),  # 30, 60, 120 km at 5 km / px
        tau: float = 1.0 / 64.0,
        soften_true: bool = False,
        threshold_weights: list[float] | None = None,
        scale_weights: list[float] | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        assert len(thresholds) >= 1, "Need at least one threshold"
        assert len(window_sizes) >= 1, "Need at least one window size"
        assert tau > 0.0, f"tau must be > 0, got: {tau}"
        assert all(
            thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)
        ), f"Thresholds must be strictly increasing, got: {thresholds}"
        assert all(
            w >= 1 for w in window_sizes
        ), f"Invalid window sizes: {window_sizes}"

        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32),
        )

        if threshold_weights is None:
            threshold_weights = [1.0] * len(thresholds)
        if scale_weights is None:
            scale_weights = [1.0] * len(window_sizes)

        assert len(threshold_weights) == len(thresholds), (
            f"threshold_weights must have len={len(thresholds)}, "
            f"got {len(threshold_weights)}"
        )
        assert len(scale_weights) == len(window_sizes), (
            f"scale_weights must have len={len(window_sizes)}, "
            f"got {len(scale_weights)}"
        )

        threshold_weights = torch.tensor(threshold_weights, dtype=torch.float32)
        scale_weights = torch.tensor(scale_weights, dtype=torch.float32)

        threshold_weights = threshold_weights / threshold_weights.sum()
        scale_weights = scale_weights / scale_weights.sum()

        self.register_buffer("threshold_weights", threshold_weights)
        self.register_buffer("scale_weights", scale_weights)

        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.tau = float(tau)
        self.soften_true = bool(soften_true)
        self.eps = float(eps)

    def _cumulative_masks(self, x: torch.Tensor, soften: bool) -> torch.Tensor:
        """
        x: [B, T, 1, H, W] in [0, 1]
        returns: [B, T, K, H, W]
        """
        thr = self.thresholds.view(1, 1, -1, 1, 1)

        if soften:
            return torch.sigmoid((x - thr) / self.tau)

        return (x >= thr).to(x.dtype)

    def _fss_per_time(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred, y_true: [B, T, 1, H, W]
        returns: [T]
        """
        B, T, C, H, W = y_pred.shape
        assert C == 1, f"Support only one output channel (tcc), got: {C}"

        y_pred = y_pred.to(torch.float32).clamp(0.0, 1.0)
        y_true = y_true.to(torch.float32).clamp(0.0, 1.0)

        # [B, T, K, H, W]
        m_pred = self._cumulative_masks(y_pred, soften=True)
        m_true = self._cumulative_masks(y_true, soften=self.soften_true)

        _, _, K, _, _ = m_pred.shape

        # [B*T, K, H, W]
        m_pred = m_pred.reshape(B * T, K, H, W)
        m_true = m_true.reshape(B * T, K, H, W)

        per_scale_loss_t = []

        for w in self.window_sizes:
            assert (
                H >= w and W >= w
            ), f"Window size {w} too large for field size {(H, W)}"

            # Valid-window pooling, no padding
            p_pred = F.avg_pool2d(m_pred, kernel_size=w, stride=1)
            p_true = F.avg_pool2d(m_true, kernel_size=w, stride=1)

            # [B*T, K]
            num = (p_pred - p_true).pow(2).sum(dim=(-2, -1))
            den = (p_pred.pow(2) + p_true.pow(2)).sum(dim=(-2, -1)).clamp_min(self.eps)

            # This is (1 - FSS) per threshold
            loss_bt_k = num / den

            # Weighted average over thresholds -> [B*T]
            loss_bt = (loss_bt_k * self.threshold_weights.view(1, K)).sum(dim=1)

            # Mean over batch -> [T]
            loss_t = loss_bt.view(B, T).mean(dim=0)
            per_scale_loss_t.append(loss_t)

        # [S, T]
        per_scale_loss_t = torch.stack(per_scale_loss_t, dim=0)

        # Weighted average over scales -> [T]
        loss_t = (per_scale_loss_t * self.scale_weights.view(-1, 1)).sum(dim=0)

        return loss_t

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        y_pred = y_pred_full
        y_true = y_true_full

        if y_true.dim() == 4:  # [B, C, H, W] -> [B, 1, C, H, W]
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        loss_t = self._fss_per_time(y_pred, y_true)  # [T]
        fss_loss = loss_t.mean()

        assert torch.isfinite(fss_loss), f"Non-finite loss: {fss_loss}"

        return {
            "loss": fss_loss,
        }
