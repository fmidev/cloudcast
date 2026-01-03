import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import lightning as L
import json
import os
import sys
from datetime import datetime, timedelta
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingLR,
    LambdaLR,
)
from common.util import (
    get_rank,
    get_latest_run_dir,
    find_latest_checkpoint_path,
    strip_prefix,
)
from typing import Optional, Callable


class cc2module(L.LightningModule):
    def __init__(
        self,
        history_length: int = 2,
        hidden_dim: int = 96,
        patch_size: int = 4,
        encoder1_depth: int = 2,
        encoder2_depth: int = 2,
        decoder1_depth: int = 2,
        decoder2_depth: int = 2,
        window_size: int = 8,
        window_size_deep: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float | list[float] = 0.0,
        learning_rate: float = 1e-3,
        warmup_iterations: int = 1000,
        weight_decay: float = 1e-3,
        rollout_length: int = 1,
        branch_from_run: str = None,
        use_gradient_checkpointing: bool = False,
        model_family: str = "pgu",
        use_scheduled_sampling: bool = False,
        ss_pred_min: float = 0.0,
        ss_pred_max: float = 1.0,
        autoregressive_mode: bool = True,
        freeze_layers: list[str] = [],
        loss_fn: Callable | nn.Module | None = None,
        use_deep_refinement_head: bool = False,
        use_hard_skip: bool = False,
        test_output_directory: str | None = None,
        use_rollout_weighting: bool = False,
        use_statistics_from_checkpoint: bool = True,
        use_residual_adapter_head: bool = False,
        use_residual_io_adapter: bool = False,
        force_frozen_backbone_to_eval: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_configured = False

        assert (use_residual_adapter_head and use_residual_io_adapter == False) or (
            use_residual_adapter_head == False
        )

        # Extract only model-specific parameters
        model_kwargs = {
            k: v
            for k, v in self.hparams.items()
            if k
            in [
                "history_length",
                "hidden_dim",
                "patch_size",
                "encoder1_depth",
                "encoder2_depth",
                "decoder1_depth",
                "decoder2_depth",
                "window_size",
                "window_size_deep",
                "num_heads",
                "mlp_ratio",
                "drop_rate",
                "attn_drop_rate",
                "drop_path_rate",
                "use_gradient_checkpointing",
                "use_scheduled_sampling",
                "ss_pred_min",
                "ss_pred_max",
                "autoregressive_mode",
                "use_deep_refinement_head",
                "use_hard_skip",
                "use_residual_adapter_head",
                "use_residual_io_adapter",
            ]
        }

        self.model_kwargs = model_kwargs

        self.test_predictions = []
        self.test_truth = []
        self.test_dates = []

        self.latest_val_tendencies = None
        self.latest_val_predictions = None
        self.latest_val_data = None

        self.latest_train_tendencies = None
        self.latest_train_predictions = None
        self.latest_train_data = None

        self.use_scheduled_sampling = use_scheduled_sampling
        self.use_statistics_from_checkpoint = use_statistics_from_checkpoint
        self.ss_pred_min = ss_pred_min
        self.ss_pred_max = ss_pred_max
        self.autoregressive_mode = autoregressive_mode
        self.test_output_directory = test_output_directory

        self._loss_fn = loss_fn

        assert self._loss_fn is not None

    def configure_model(self) -> None:
        if self.model_configured:
            return

        # Read input_resolution from data
        self.input_resolution = self.trainer.datamodule.input_resolution
        rank_zero_info(f"Data resolution is {self.input_resolution}")

        self.model_kwargs.update(
            {
                "input_resolution": self.trainer.datamodule.input_resolution,
                "prognostic_params": self.trainer.datamodule.hparams.prognostic_params,
                "forcing_params": self.trainer.datamodule.hparams.forcing_params,
                "static_forcing_params": self.trainer.datamodule.hparams.static_forcing_params,
            }
        )

        self._build_model()

        self.model_configured = True

    def _store_trainable_module_names(self) -> None:
        trainable_modules = set()
        for module_name, module in self.model.named_modules():
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    trainable_modules.add(module_name)
                    break

        self._trainable_module_names = trainable_modules
        rank_zero_info("Backbone will be set to eval to disable dropouts")

    def _force_frozen_backbone_to_eval(self) -> None:
        if not hasattr(self, "_trainable_module_names"):
            return

        # 1) everything deterministic
        self.model.eval()

        # 2) re-enable training behavior only on modules that own trainable params
        for module_name, module in self.model.named_modules():
            if module_name in self._trainable_module_names:
                module.train()

    def _print_trainable_layers(self):
        trainable = set()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.add(name)

        rank_zero_info(f"Trainable layers: {trainable}")

    def _freeze_layers(self) -> None:
        if len(self.hparams.freeze_layers) == 0:
            return

        frozen = []

        for l in self.hparams.freeze_layers:
            # start with submodules
            for name, module in self.model.named_modules():
                if name.startswith(l):
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(name)

            # also direct parameters on the model (not in submodules)
            for name, param in self.model.named_parameters(recurse=False):
                if name.startswith(l):
                    param.requires_grad = False
                    frozen.append(name)

        rank_zero_info(f"Froze layers: {frozen}")
        self._store_trainable_module_names()
        self._print_trainable_layers()

    def _build_model(self) -> None:
        if self.hparams.model_family == "pgu":
            from pgu.cc2 import cc2model

            if self.hparams.autoregressive_mode:
                from pgu.util import roll_forecast
            else:
                from pgu.util import roll_forecast_direct as roll_forecast

        elif self.hparams.model_family == "swinu":
            from swinu.cc2 import cc2model
            from swinu.util import roll_forecast

        self.model_class = self.hparams.model_family
        self._roll_forecast = roll_forecast

        self.model = cc2model(config=self.model_kwargs)

        self.run_dir = None
        if self.trainer.state.stage in ("train", "test"):
            self.run_name = os.environ["CC2_RUN_NAME"]
            self.run_number = int(os.environ.get("CC2_RUN_NUMBER", -1))
            self.run_dir = os.environ["CC2_RUN_DIR"]

        if self.hparams.branch_from_run:
            if "/" in self.hparams.branch_from_run:
                branch_run_dir = "runs/{}".format(self.hparams.branch_from_run)
            else:
                branch_run_dir = get_latest_run_dir(
                    "runs/" + self.hparams.branch_from_run
                )

            ckpt_path = find_latest_checkpoint_path(branch_run_dir)

            rank_zero_info(
                f"Branching from {branch_run_dir} using weights from: {ckpt_path}"
            )

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            state_dict = ckpt["state_dict"]
            state_dict = strip_prefix(state_dict)

            # Load the state dict
            # strict=False allows missing/extra keys (e.g., different final layer)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            rank_zero_info(f"Weight loading results: {load_result}")

        self._freeze_layers()

        rank = get_rank()

        msg = "Rank {} starting at {}".format(get_rank(), datetime.now())

        if self.run_dir is not None:
            msg += f" using run directory {self.run_dir}"

        print(msg)

    def on_train_batch_start(self, batch, batch_idx):
        if getattr(self.hparams, "force_frozen_backbone_to_eval", False):
            self._force_frozen_backbone_to_eval()

    def on_train_start(self) -> None:
        self.max_epochs = self.trainer.max_epochs
        # max_steps is not typically set (by me)
        if self.trainer.max_steps > 0:
            self.max_steps = self.trainer.max_steps
        else:
            # fallback if only max_epochs is set
            self.max_steps = self.trainer.estimated_stepping_batches

        assert self.max_steps > 0, "Trainer must be configured with max_steps"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # data, forcing, step)

    def training_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = self._roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self._loss_fn,
            use_scheduled_sampling=self.use_scheduled_sampling,
            step=self.global_step,
            max_step=self.max_steps,
            ss_pred_min=self.ss_pred_min,
            ss_pred_max=self.ss_pred_max,
            use_rollout_weighting=self.hparams.use_rollout_weighting,
        )

        self.log("train_loss", loss["loss"], sync_dist=False)

        for k, v in loss.items():
            if isinstance(v, torch.Tensor):
                v = torch.sum(v).item()

            self.log(
                f"loss/train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, sync_dist=False)

        if batch_idx == 0:
            self.latest_train_tendencies = tendencies.float()
            self.latest_train_predictions = predictions.float()
            self.latest_train_data = data

        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def validation_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = self._roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self._loss_fn,
            use_scheduled_sampling=False,
            use_rollout_weighting=self.hparams.use_rollout_weighting,
        )

        self.log("val_loss", loss["loss"], sync_dist=False)

        for k, v in loss.items():
            if isinstance(v, torch.Tensor):
                v = torch.sum(v).item()

            self.log(
                f"loss/val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        if batch_idx == 0:
            self.latest_val_tendencies = tendencies.float()
            self.latest_val_predictions = predictions.float()
            self.latest_val_data = data

        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def test_step(self, batch, batch_idx):
        data, forcing, dates = batch

        _, tendencies, predictions = self._roll_forecast(
            self,
            data,
            forcing,
            self.hparams.rollout_length,  # Access from hparams
            loss_fn=None,
            use_scheduled_sampling=False,
            use_rollout_weighting=self.hparams.use_rollout_weighting,
        )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)

        predictions = torch.concatenate((analysis_time, predictions), dim=1)
        truth = torch.concatenate((analysis_time, data[1]), dim=1)

        self.test_predictions.append(predictions)
        self.test_truth.append(truth)
        self.test_dates.append(torch.concatenate((dates[0][:, -1:], dates[1]), dim=1))

        return {
            "tendencies": tendencies,
            "predictions": predictions,
            "source": data[0],
            "truth": data[1],
        }

    def predict_step(self, batch, batch_idx):
        data, forcing, dates = batch

        _, tendencies, predictions = self._roll_forecast(
            self,
            data,
            forcing,
            self.hparams.rollout_length,  # Access from hparams
            loss_fn=None,
            use_scheduled_sampling=False,
            use_rollout_weighting=False,
        )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)

        predictions = torch.concatenate((analysis_time, predictions), dim=1)

        self.test_predictions.append(predictions)
        self.test_dates.append(torch.concatenate((dates[0][:, -1:], dates[1]), dim=1))

        return {
            "tendencies": tendencies,
            "predictions": predictions,
            "source": data[0],
            "truth": data[1],
        }

    def _gather(self, predictions, truth, dates):
        # Gather across all DDP ranks
        predictions = self.all_gather(predictions)

        # all_gather adds a new dimension [world_size, ...], so reshape
        predictions = predictions.reshape(-1, *predictions.shape[2:])

        dates = self.all_gather(dates)

        # Dates is 2D [batch, time], so flatten only first two dims
        dates = dates.reshape(-1, dates.shape[-1])

        # Sort by dates to restore chronological order
        sort_idx = torch.argsort(dates[:, 0])
        predictions = predictions[sort_idx]
        dates = dates[sort_idx]

        if len(truth):
            truth = self.all_gather(truth)
            truth = truth.reshape(-1, *truth.shape[2:])
            truth = truth[sort_idx]

        return predictions, truth, dates

    def _write_results_on_test_predict_end(self):
        def _write(tensor, filename):
            torch.save(tensor, filename)
            print(f"Wrote file {filename} (shape: {tensor.shape})")

        # Get the run directory from the checkpoint path
        if self.test_output_directory is None:
            run_dir = self.run_dir if self.run_dir is not None else "."
            self.test_output_directory = f"{run_dir}/test-output"

        os.makedirs(self.test_output_directory, exist_ok=True)

        predictions = torch.concatenate(self.test_predictions)
        dates = torch.concatenate(self.test_dates)

        truth = []
        if len(self.test_truth):
            truth = torch.concatenate(self.test_truth)

        if self.trainer.world_size > 1:
            predictions, truth, dates = self._gather(predictions, truth, dates)

        # Only rank 0 writes to avoid duplicate writes
        if self.trainer.is_global_zero:
            _write(predictions, f"{self.test_output_directory}/predictions.pt")

            if len(truth):
                _write(truth, f"{self.test_output_directory}/truth.pt")

            _write(dates, f"{self.test_output_directory}/dates.pt")

    def on_test_end(self):
        self._write_results_on_test_predict_end()

    def on_predict_end(self):
        self._write_results_on_test_predict_end()

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "pos_embed", "norm"]):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
        )

        if self.trainer.max_epochs > 0:
            effective_batch_size = (
                (self.trainer.num_devices * self.trainer.num_nodes)  # Total devices
                * self.trainer.accumulate_grad_batches
                * self.trainer.datamodule.hparams.batch_size  # Get batch size from datamodule hparams
            )

            train_ds_size = len(self.trainer.datamodule.ds_train)
            steps_per_epoch = (
                train_ds_size + effective_batch_size - 1
            ) // effective_batch_size

            num_iterations = steps_per_epoch * self.trainer.max_epochs
            T_max = num_iterations - self.hparams.warmup_iterations

        elif self.trainer.max_steps is not None:
            assert (
                self.trainer.max_steps > self.hparams.warmup_iterations
            ), f"max_steps {self.trainer.max_steps} is smaller than warmup_iterations {self.hparams.warmup_iterations}"

            T_max = self.trainer.max_steps - self.hparams.warmup_iterations

        else:
            raise ValueError("Either max_epochs or max_steps needs to be defined")

        assert T_max > 0, f"max iterations is {T_max}"

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-7,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_iterations,
        )

        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        dm = self.trainer.datamodule
        if dm and dm._full_dataset and hasattr(dm._full_dataset, "data"):
            checkpoint["data_statistics"] = dm._full_dataset.data.statistics
            checkpoint["data_statististics_name_to_index"] = (
                dm._full_dataset.data.name_to_index
            )
        else:
            rank_zero_warn("Statistics not saved to checkpoint")
