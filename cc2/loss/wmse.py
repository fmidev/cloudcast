import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, d0: float = 0.1, alpha: float = 1.0, w_max: float = 3.0):
        super().__init__()
        self.d0 = d0
        self.alpha = alpha
        self.w_max = w_max

    def forward(self, y_pred_delta: torch.Tensor, y_true_delta: torch.Tensor, **kwargs):
        """
        Weighted MSE loss.

        Weights are larger where |y_true| is large (i.e. where the true delta
        is large), to emphasise dynamically active pixels.

        Args:
            y_true: [B, C, H, W] tensor of targets (deltas or absolute values).
            y_pred: [B, C, H, W] tensor of predictions.
            d0: scale for |y_true| at which weighting starts to ramp up.
            alpha: strength of additional weighting.
            w_max: maximum weight (to avoid a few pixels dominating).

        Returns:
            dict with total loss and the weighted MSE scalar.
        """
        y_true = y_true_full
        y_pred = y_pred_full

        residual = y_pred - y_true  # [B,C,H,W]
        abs_delta = y_true.abs()

        # Raw weights: 1 + alpha * (|Δ| / d0)
        weights = 1.0 + self.alpha * (abs_delta / self.d0)
        weights = torch.clamp(weights, 1.0, self.w_max)

        # Normalise to keep overall scale comparable to plain MSE
        weights = weights / weights.mean().detach()

        loss = (weights * residual**2).mean()

        assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
        return {"loss": loss, "mse_loss": loss}
