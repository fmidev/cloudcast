import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self, use_full_state: bool = False):
        super().__init__()
        self.use_full_state = use_full_state

    def forward(
        self,
        y_pred_delta: torch.Tensor,
        y_true_delta: torch.Tensor,
        y_pred_full: torch.Tensor,
        y_true_full: torch.Tensor,
        **kwargs
    ):
        if self.use_full_state:
            loss = torch.abs(y_true_full - y_pred_full).mean()
        else:
            loss = torch.abs(y_true_delta - y_pred_delta).mean()
        assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)
        return {"loss": loss}
