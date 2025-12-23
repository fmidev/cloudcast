# pgu/gm_callback.py
import torch
from lightning.pytorch.callbacks import Callback

EPS = 1e-12


def _flatten_grads(module):
    """Collect and flatten all parameter grads from a module."""
    grads = []
    for p in module.parameters(recurse=True):
        if p.grad is not None:
            g = p.grad.detach().float()
            grads.append(g.reshape(-1))
    if not grads:
        return None
    return torch.cat(grads, dim=0)


def _mean(x):
    return x.mean()


def _std(x):
    return x.std(unbiased=False)


def _rms(x):
    return (x.pow(2).mean() + EPS).sqrt()


def _max_abs(x):
    return x.abs().max()


def _frac_nan(x):
    return torch.isnan(x).float().mean()


class GradientMonitorCallback(Callback):
    """
    Auto-discovers model submodules and logs gradient stats (mean, std, rms, max_abs, frac_nan)
    plus a global grad norm. Mirrors your activation logger style.

    Args:
      log_every_n_steps: emit logs every N training steps (aggregated as epoch metrics).
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.modules = {}  # name -> nn.Module (populated at fit start)
        self.buffer = {}  # name -> stats dict
        self.global_grad_norm = None

    def _discover_modules(self, pl_module):
        m = getattr(pl_module, "model", pl_module)  # handle LightningModule wrappers

        def add(name, obj):
            if obj is not None:
                self.modules[name] = obj

        # singletons (if present)
        add("patch_embed", getattr(m, "patch_embed", None))
        add("downsample", getattr(m, "downsample", None))
        add("upsample", getattr(m, "downsample", None))
        add("patch_expand", getattr(m, "patch_expand", None))
        add("final_expand", getattr(m, "final_expand", None))

        # all blocks inside these lists (if they exist)
        for stage in ["encoder1", "encoder2", "decoder1", "decoder2"]:
            lst = getattr(m, stage, None)
            if lst is None:
                continue
            for i, block in enumerate(lst):
                add(f"{stage}.{i}", block)

    def on_fit_start(self, trainer, pl_module):
        # build module map once
        self.modules.clear()
        self._discover_modules(pl_module)

    def on_after_backward(self, trainer, pl_module):
        # collect per-module grad stats
        self.buffer.clear()
        for name, mod in self.modules.items():
            g = _flatten_grads(mod)
            if g is None or g.numel() == 0:
                continue
            self.buffer[name] = {
                "grad_mean": float(_mean(g)),
                "grad_std": float(_std(g)),
                "grad_rms": float(_rms(g)),
                "grad_max_abs": float(_max_abs(g)),
                "grad_frac_nan": float(_frac_nan(g)),
            }

        # global grad norm over all params
        total_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().float().pow(2).sum())
        self.global_grad_norm = float(total_sq**0.5)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        # emit as epoch metrics (mirrors your activation callback)
        for name, stats in self.buffer.items():
            for k, v in stats.items():
                pl_module.log(
                    f"grads/{name}_{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )
        if self.global_grad_norm is not None:
            pl_module.log(
                "grads/global_grad_norm",
                self.global_grad_norm,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
        # clear buffers
        self.buffer.clear()
        self.global_grad_norm = None
