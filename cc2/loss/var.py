import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_btchw(x: torch.Tensor) -> torch.Tensor:
    """
    Accept [B,T,C,H,W] or [B,C,H,W] or [B,H,W] and return [B,T,C,H,W].
    """
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        return x[:, None, ...]  # add T
    if x.ndim == 3:
        return x[:, None, None, ...]  # add T and C
    raise ValueError(f"Expected 3/4/5D tensor, got {tuple(x.shape)}")


def _highpass_2d(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    High-pass via local mean removal:
        hp = x - avg_pool(x, k)
    x is [N,C,H,W].
    """
    if k <= 1:
        return x
    pad = k // 2
    # reflect padding avoids edge darkening compared to zero-pad
    xpad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    blur = F.avg_pool2d(xpad, kernel_size=k, stride=1)
    # blur is back to [N,C,H,W]
    return x - blur


class VarianceLoss(nn.Module):
    """
    High-pass variance matching loss.

    Loss:
      For each lead t and each hp kernel k:
        vp = Var(hp(pred))
        vt = Var(hp(true))
      Compare in log-space for scale stability:
        L = mean_t,k | log(vp + eps) - log(vt + eps) |
    """

    def __init__(
        self,
        ks: tuple[int, ...] = (5, 11, 21),
        eps: float = 1e-6,
        reduce_over: tuple[int, ...] = (0, 2, 3),  # var over (B,H,W), keep C if C>1
    ):
        super().__init__()
        self.ks = ks
        self.eps = eps
        self.reduce_over = reduce_over

    def forward(
        self,
        y_pred_full: torch.Tensor,
        y_true_full: torch.Tensor,
        **kwargs,
    ):
        yp = _ensure_btchw(y_pred_full)
        yt = _ensure_btchw(y_true_full)
        assert (
            yp.shape == yt.shape
        ), f"Shape mismatch: {tuple(yp.shape)} vs {tuple(yt.shape)}"

        B, T, C, H, W = yp.shape

        # fold (B,T) -> N for filtering: [N,C,H,W]
        yp2 = yp.reshape(B * T, C, H, W).to(torch.float32)
        yt2 = yt.reshape(B * T, C, H, W).to(torch.float32)

        losses = []
        for k in self.ks:
            hp_p = _highpass_2d(yp2, k)
            hp_t = _highpass_2d(yt2, k)

            # variance per-channel (C), then mean over channels to get scalar
            vp_c = hp_p.var(
                unbiased=False, dim=self.reduce_over
            )  # -> [C] if reduce_over excludes C
            vt_c = hp_t.var(unbiased=False, dim=self.reduce_over)

            vp = vp_c.mean()
            vt = vt_c.mean()

            # log-energy match (more stable than raw ratio)
            lk = torch.abs(torch.log(vp + self.eps) - torch.log(vt + self.eps))
            losses.append(lk)

        loss = torch.stack(losses).mean()

        assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
        return {"loss": loss}
