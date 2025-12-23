import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    losses: sequence of instantiated loss modules (nn.Module)
    weights: same length, per-loss scalar weights
    names: optional list of names for logging; otherwise infer from class names
    """

    def __init__(
        self,
        losses: list[nn.Module],
        weights: list[float] | None = None,
        names: list[str] | None = None,
    ):
        super().__init__()

        if weights is None:
            weights = [1.0] * len(losses)
        assert len(losses) == len(weights), "losses and weights must have same length"

        if names is None:
            names = [l.__class__.__name__ for l in losses]
        assert len(names) == len(losses), "names and losses must have same length"

        self.loss_modules = nn.ModuleList(losses)
        self.loss_weights = list(weights)
        self.loss_names = list(names)

    def forward(self, **kwargs):
        """
        Pass named tensors (delta_pred, delta_true, y_pred_full, y_true_full, ...)
        All sub-losses receive the same kwargs and pick what they need.
        """
        total = 0.0
        components: dict[str, torch.Tensor] = {}

        for weight, name, module in zip(
            self.loss_weights, self.loss_names, self.loss_modules
        ):
            out = module(**kwargs)

            if isinstance(out, dict):
                value = out["loss"]
                for k, v in out.items():
                    components[f"{name}_{k}"] = v
            else:
                value = out
                components[f"{name}"] = value

            total = total + weight * value

        components["loss"] = total
        return components
