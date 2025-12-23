import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIMLoss(nn.Module):
    def __init__(self, sigma: float = 1.5, kernel_size: int = 11):
        super().__init__()

        self.ssim = StructuralSimilarityIndexMeasure(
            gaussian_kernel=True,
            sigma=sigma,
            kernel_size=kernel_size,
            data_range=1.0,
            reduction="elementwise_mean",
        )

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        if y_true_full.dim() == 4:
            y_true = y_true_full
            y_pred = y_pred_full
        elif y_true_full.dim() == 5:
            B, T, C, H, W = y_pred_full.shape
            y_pred = y_pred_full.view(B * T, C, H, W)
            y_true = y_true_full.view(B * T, C, H, W)

        ssim_val = self.ssim(y_pred, y_true)
        ssim_loss = 1.0 - ssim_val

        assert torch.isfinite(ssim_loss), f"Non-finite loss: {ssim_loss}"

        loss = {
            "loss": ssim_loss,
        }

        return loss
