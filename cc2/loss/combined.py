import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class RampConfig:
    """
    Piecewise-linear ramp schedule for a selected loss component.

    Example:
      - warmup_steps=10_000 (weight value 0)
      - ramp_steps=20_000   (linearly increase 0->1)
      - after warmup+ramp:  (weight value 1)
    """

    warmup_steps: int = 10_000
    ramp_steps: int = 20_000

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        t = (step - self.warmup_steps) / float(self.ramp_steps)
        return float(max(0.0, min(1.0, t)))


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
        ramps: list[None | RampConfig] | None = None,
        names: list[str] | None = None,
    ):
        super().__init__()

        if weights is None:
            weights = [1.0] * len(losses)
        assert len(losses) == len(weights), "losses and weights must have same length"

        if ramps is not None and len(ramps) != len(losses):
            raise ValueError("ramps must be None or have same length as losses")

        if names is None:
            names = [l.__class__.__name__ for l in losses]
        assert len(names) == len(losses), "names and losses must have same length"

        self.loss_modules = nn.ModuleList(losses)
        self.loss_weights = list(weights)
        self.loss_ramps = list(ramps) if ramps is not None else [None] * len(weights)
        self.loss_names = list(names)

        self._have_any_ramp = any(r is not None for r in self.loss_ramps)

    def forward(self, **kwargs):
        """
        Pass named tensors (delta_pred, delta_true, y_pred_full, y_true_full, ...)
        All sub-losses receive the same kwargs and pick what they need.
        """
        if self._have_any_ramp:
            if "global_step" not in kwargs:
                raise ValueError(
                    "CombinedLoss: global_step is required when ramps are enabled"
                )
            gs = kwargs.get("global_step", 0)
            step = int(gs) if gs is not None else 0  # None during sanity check
        else:
            step = 0

        total = 0.0
        components: dict[str, torch.Tensor] = {}

        for i, (base_weight, name, module) in enumerate(
            zip(self.loss_weights, self.loss_names, self.loss_modules)
        ):
            out = module(**kwargs)

            if isinstance(out, dict):
                value = out["loss"]
                for k, v in out.items():
                    components[f"{name}_{k}"] = v
            else:
                value = out
                components[f"{name}"] = value

            weight = float(base_weight)

            rc = self.loss_ramps[i]
            if rc is not None:
                rf = rc.factor(step)
                weight *= rf
                components[f"{name}_ramp_factor"] = torch.as_tensor(
                    rf, device=value.device, dtype=value.dtype
                )

            total += weight * value

        components["loss"] = total
        return components
