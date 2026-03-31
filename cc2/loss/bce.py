import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary cross-entropy loss treating TCC as a probability in [0, 1].

    Unlike MAE/MSE, BCE is infinite at the boundaries for wrong predictions,
    which prevents the model from collapsing to hard 0/1 outputs when the
    target has intermediate values (broken cloud).

    Targets and predictions are clipped to [eps, 1-eps] before the log to
    avoid numerical issues at exact 0 and 1.
    """

    def __init__(self, use_full_state: bool = True, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred_full: torch.Tensor,
        y_true_full: torch.Tensor,
        **kwargs,
    ):
        p = y_pred_full.float()
        y = y_true_full.float()
        p = p.clamp(self.eps, 1.0 - self.eps)
        y = y.clamp(self.eps, 1.0 - self.eps)

        # F.binary_cross_entropy is unsafe under autocast; convert to logits
        # and use the autocast-safe variant instead (mathematically equivalent)
        loss = F.binary_cross_entropy_with_logits(torch.logit(p), y, reduction="mean")

        assert torch.isfinite(loss), f"Non-finite BCE loss: {loss}"
        return {"loss": loss}
