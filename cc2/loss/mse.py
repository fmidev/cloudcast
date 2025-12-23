import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred_delta: torch.Tensor, y_true_delta: torch.Tensor, **kwargs):
        loss = ((y_true_delta - y_pred_delta) ** 2).mean()
        assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)
        return {"loss": loss, "mse_loss": loss}
