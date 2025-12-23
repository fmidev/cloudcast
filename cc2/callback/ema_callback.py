from __future__ import annotations
from copy import deepcopy
import torch
import lightning as L


def _ema_update_(ema_model: torch.nn.Module, src_model: torch.nn.Module, decay: float):
    with torch.no_grad():
        for (k, ema_p), (_, src_p) in zip(
            ema_model.named_parameters(), src_model.named_parameters()
        ):
            src = src_p.detach()
            if src.device != ema_p.device or src.dtype != ema_p.dtype:
                src = src.to(device=ema_p.device, dtype=ema_p.dtype)
            ema_p.mul_(decay).add_(src, alpha=1.0 - decay)

        for (k, ema_b), (_, src_b) in zip(
            ema_model.named_buffers(), src_model.named_buffers()
        ):
            if ema_b.device != src_b.device or ema_b.dtype != src_b.dtype:
                ema_b.copy_(src_b.to(device=ema_b.device, dtype=ema_b.dtype))
            else:
                ema_b.copy_(src_b)


class EMACallback(L.Callback):
    def __init__(
        self,
        decay: float = 0.999,
        start_step: int = 0,
        evaluate_ema: bool = True,
        store_on_cpu: bool = True,
    ):
        """
        decay:        smoothing factor
        start_step:   start updating EMA after this global_step
        evaluate_ema: if True, swap to EMA for val/test/predict automatically
        store_on_cpu: keep EMA weights in fp32 on CPU (safer with AMP/bf16)
        """
        self.decay = decay
        self.start_step = start_step
        self.evaluate_ema = evaluate_ema
        self.store_on_cpu = store_on_cpu

        self.ema_model: torch.nn.Module | None = None
        self._backup_sd: dict | None = None  # to restore after eval

    # ----- lifecycle -----
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        dev = next(pl_module.parameters()).device
        self.ema_model = deepcopy(pl_module).to(device=dev, dtype=torch.float32)
        self.ema_model.requires_grad_(False)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        if self.ema_model is None or trainer.global_step < self.start_step:
            return
        src_model = pl_module
        if next(pl_module.parameters()).dtype != torch.float32:
            src_model = pl_module.float()
        _ema_update_(self.ema_model, src_model, self.decay)

    def _swap_in_ema(self, pl_module: L.LightningModule):
        if not self.evaluate_ema or self.ema_model is None:
            return
        # stash current weights
        self._backup_sd = {
            k: v.detach().cpu() for k, v in pl_module.state_dict().items()
        }
        # load EMA weights to the training device/dtype
        ema_sd = self.ema_model.state_dict()
        pl_module.load_state_dict(ema_sd, strict=True)

    def _restore_from_backup(self, pl_module: L.LightningModule):
        if self._backup_sd is None:
            return
        pl_module.load_state_dict(self._backup_sd, strict=True)
        self._backup_sd = None

    def on_validation_start(self, trainer, pl_module):
        self._swap_in_ema(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._restore_from_backup(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_in_ema(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_from_backup(pl_module)

    def on_predict_start(self, trainer, pl_module):
        self._swap_in_ema(pl_module)

    def on_predict_end(self, trainer, pl_module):
        self._restore_from_backup(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_model is None:
            return
        ema_sd = {k: v.detach().cpu() for k, v in self.ema_model.state_dict().items()}
        checkpoint["ema_state_dict"] = ema_sd
        checkpoint["ema_decay"] = self.decay
        checkpoint["ema_start_step"] = self.start_step

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        ema_sd = checkpoint.get("ema_state_dict", None)
        if ema_sd is None:
            return
        print("Loading EMA checkpoint")
        if self.ema_model is None:
            dev = next(pl_module.parameters()).device
            self.ema_model = deepcopy(pl_module).to(device=dev, dtype=torch.float32)
            self.ema_model.requires_grad_(False)
        self.ema_model.load_state_dict(ema_sd, strict=True)
