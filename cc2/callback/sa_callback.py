# stage_activation_mlflow_callback.py
import torch
from lightning.pytorch.callbacks import Callback


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for t in x:
            if torch.is_tensor(t):
                return t
    return None


def _flatten_for_stats(t: torch.Tensor) -> torch.Tensor:
    # flatten per-sample features -> [B, N]
    if t.ndim >= 2:
        return t.flatten(1)
    return t.view(t.shape[0], -1)


def _rms(x: torch.Tensor) -> torch.Tensor:
    x = _flatten_for_stats(x)
    return (torch.norm(x, dim=1) / (x.shape[1] ** 0.5)).mean()


def _mean(x: torch.Tensor) -> torch.Tensor:
    return _flatten_for_stats(x).mean()


def _std(x: torch.Tensor) -> torch.Tensor:
    return _flatten_for_stats(x).std(unbiased=False)


def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    # q in [0,100]
    x = _flatten_for_stats(x)
    k = max(1, int((q / 100.0) * (x.numel() - 1)))
    return x.view(-1).kthvalue(k).values


def _max_abs(x: torch.Tensor) -> torch.Tensor:
    return _flatten_for_stats(x).abs().max()


def _frac_nan(x: torch.Tensor) -> torch.Tensor:
    x = _flatten_for_stats(x)
    return torch.isnan(x).float().mean()


class StageActivationCallback(Callback):
    """
    Logs activation statistics to MLflow during training/validation.
    Hooks: patch_embed, downsample, and all blocks in encoder1/2, decoder1/2.
    No stdout.
    """

    def __init__(self, log_every_n_batches: int = 100):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.hooks = []
        self.buffer = {}  # name -> dict(stats)

    def _hook(self, name):
        # forward_hook(module, inputs, output)
        def fn(module, args, out):
            x_out = _first_tensor(out)
            x_in = (
                _first_tensor(args[0])
                if isinstance(args, (tuple, list)) and args
                else None
            )
            if x_out is None:
                return
            with torch.no_grad():
                stats = {}
                stats["rms_out"] = _rms(x_out).item()
                stats["mean"] = _mean(x_out).item()
                stats["std"] = _std(x_out).item()
                stats["max_abs"] = _max_abs(x_out).item()
                stats["frac_nan"] = _frac_nan(x_out).item()
                if x_in is not None:
                    stats["rms_in"] = _rms(x_in).item()
                    denom = max(stats["rms_in"], 1e-12)
                    stats["gain"] = stats["rms_out"] / denom
                self.buffer[name] = stats

        return fn

    def _reg(self, name, module):
        if module is None:
            return
        self.hooks.append(module.register_forward_hook(self._hook(name)))

    def on_fit_start(self, trainer, pl_module):
        m = getattr(pl_module, "model", pl_module)

        # Single modules
        self._reg("patch_embed", getattr(m, "patch_embed", None))
        self._reg("downsample", getattr(m, "downsample", None))
        self._reg("upsample", getattr(m, "upsample", None))
        self._reg("patch_expand", getattr(m, "patch_expand", None))
        self._reg("final_expand", getattr(m, "final_expand", None))
        self._reg("post_pe_norm", getattr(m, "post_pe_norm", None))
        self._reg("post_merge_norm", getattr(m, "post_merge_norm", None))
        self._reg("pre_dec1_norm", getattr(m, "pre_dec1_norm", None))
        self._reg("pre_dec2_norm", getattr(m, "pre_dec2_norm", None))
        self._reg("pre_dec2_norm_skip", getattr(m, "pre_dec2_norm_skip", None))
        self._reg("post_fuse_norm", getattr(m, "post_fuse_norm", None))

        # All blocks within ModuleLists
        for stage in ["encoder1", "encoder2", "decoder1", "decoder2"]:
            lst = getattr(m, stage, None)
            if lst is None:
                continue
            for i, block in enumerate(lst):
                self._reg(f"{stage}.{i}", block)

    def on_fit_end(self, trainer, pl_module):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _emit_mlflow(self, pl_module, prefix="train"):
        if not self.buffer:
            return
        # flatten metrics as prefix + name + / + stat
        for name, d in self.buffer.items():
            for k, v in d.items():
                pl_module.log(
                    f"activations/{prefix}/{name}_{k}",
                    float(v),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )

        self.buffer.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.log_every_n_batches == 0:
            self._emit_mlflow(pl_module, prefix="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # log after first val batch each epoch
        if batch_idx == 0:
            self._emit_mlflow(pl_module, prefix="val")
