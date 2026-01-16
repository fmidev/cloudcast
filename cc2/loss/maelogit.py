import torch
import torch.nn as nn


def logit_clamped(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class MAELogitLoss(nn.Module):
    """
    MAE computed in logit space:
        L = mean(|logit(y_true) - logit(y_pred)|)

    Intended for full-state supervision (use_full_state=True in your pipeline).
    Delta loss in logit space is not supported here, because logit(delta) is undefined.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred_full: torch.Tensor,
        y_true_full: torch.Tensor,
        **kwargs
    ):
        z_true = logit_clamped(y_true_full, self.eps)
        z_pred = logit_clamped(y_pred_full, self.eps)
        loss = torch.abs(z_true - z_pred).mean()

        assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
        return {"loss": loss}
