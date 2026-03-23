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
        freeze_layers: list[str] = [],
        loss_fn: nn.Module | None = None,
        test_output_directory: str | None = None,
        use_rollout_weighting: bool = False,
        use_statistics_from_checkpoint: bool = True,
        force_frozen_backbone_to_eval: bool = False,
        preprocessor: Callable | None = None,
        use_flow_matching: bool = False,
        num_inference_steps: int = 4,
        flow_warm_start: bool = True,
        warm_start_alpha: float = 0.4,
        flow_init_noise_sigma: float = 1.0,
        flow_eta: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_configured = False

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
                "use_flow_matching",
            ]
        }

        # hparams serializes values to dict, so assign preprocessor here manually
        model_kwargs["preprocessor"] = preprocessor

        self.model_kwargs = model_kwargs

        self.test_predictions = []
        self.test_truth = []
        self.test_predictions_noo = []
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

    def _extend_proj_force_for_flow(self, state_dict: dict) -> dict:
        """
        When use_flow_matching=True the model's proj_force expects 2 extra input
        channels (x_alpha and alpha_map). A checkpoint from before flow matching
        has the smaller weight. Zero-extend it so load_state_dict works.
        """
        key_w = "patch_embed.proj_force.weight"
        key_b = "patch_embed.proj_force.bias"

        if key_w not in state_dict:
            return state_dict

        old_w = state_dict[key_w]  # [d_force, Cf_old*ps*ps]
        model_w = self.model.patch_embed.proj_force.weight  # [d_force, Cf_new*ps*ps]

        if old_w.shape == model_w.shape:
            return state_dict  # already the right size

        d_force, old_in = old_w.shape
        new_in = model_w.shape[1]
        new_w = torch.zeros(d_force, new_in, dtype=old_w.dtype, device=old_w.device)
        new_w[:, :old_in] = old_w
        state_dict[key_w] = new_w

        rank_zero_info(
            f"Extended proj_force.weight from [{d_force},{old_in}] to [{d_force},{new_in}] "
            f"(zero-initialised columns for x_alpha and alpha_map channels)"
        )
        return state_dict

    def _build_model(self) -> None:
        if self.hparams.model_family == "pgu":
            from pgu.cc2 import cc2model
            from pgu.util import roll_forecast

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

            if self.hparams.use_flow_matching:
                state_dict = self._extend_proj_force_for_flow(state_dict)

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

    def _build_flow_forcing(self, forcing, y):
        """
        Corrupt the clean targets and inject as extra forcing channels for flow matching.

        forcing : [B, T+n_step, C_force, H, W]  – original forcing (no flow channels)
        y       : [B, n_step,   C_data,  H, W]  – clean future targets

        Returns: [B, T+n_step, C_force+2, H, W] where the last 2 channels at each
        future time slot carry (x_alpha, alpha_map); history slots carry zeros.
        """
        B, T_total, C_force, H, W = forcing.shape
        _, n_step, C_data, _, _ = y.shape
        T = T_total - n_step

        # Sample one alpha per sample in the batch ~ Uniform(0, 1)
        alpha = torch.rand(B, device=forcing.device)  # [B]

        # Corrupt target: x_alpha = (1-alpha)*y + alpha*z
        alpha_5d = alpha.view(B, 1, 1, 1, 1)
        z = torch.randn_like(y)
        x_alpha = (1.0 - alpha_5d) * y + alpha_5d * z  # [B, n_step, C_data, H, W]

        # Allocate extended forcing (history slots have zero flow channels)
        extra = torch.zeros(
            B, T_total, 2, H, W, device=forcing.device, dtype=forcing.dtype
        )
        for t in range(n_step):
            extra[:, T + t, 0, :, :] = x_alpha[
                :, t, 0, :, :
            ]  # x_alpha  (TCC is single-channel)
            extra[:, T + t, 1, :, :] = alpha.view(B, 1, 1).expand(B, H, W)  # alpha_map

        return torch.cat([forcing, extra], dim=2)  # [B, T+n_step, C_force+2, H, W]

    def training_step(self, batch, batch_idx):
        data, forcing = batch

        if self.hparams.use_flow_matching:
            _, y = data
            forcing = self._build_flow_forcing(forcing, y)

        loss, outs = self._roll_forecast(
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

        tendencies = outs["tendencies"]
        predictions = outs["predictions"]

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

        if self.hparams.use_flow_matching:
            _, y = data
            forcing = self._build_flow_forcing(forcing, y)

        loss, outs = self._roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self._loss_fn,
            use_scheduled_sampling=False,
            use_rollout_weighting=self.hparams.use_rollout_weighting,
        )

        tendencies = outs["tendencies"]
        predictions = outs["predictions"]

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

        if self.hparams.use_flow_matching:
            from swinu.util import flow_roll_forecast

            # Pre-allocate the 2 extra flow channels (filled during denoising)
            B, T_total, C_force, H, W = forcing.shape
            extra = torch.zeros(
                B, T_total, 2, H, W, device=forcing.device, dtype=forcing.dtype
            )
            flow_forcing = torch.cat([forcing, extra], dim=2)

            _, outs = flow_roll_forecast(
                self.model,
                data,
                flow_forcing,
                self.hparams.rollout_length,
                num_inference_steps=self.hparams.num_inference_steps,
                flow_warm_start=self.hparams.flow_warm_start,
                warm_start_alpha=self.hparams.warm_start_alpha,
                init_noise_sigma=self.hparams.flow_init_noise_sigma,
                eta=self.hparams.flow_eta,
            )
        else:
            _, outs = self._roll_forecast(
                self.model,
                data,
                forcing,
                self.hparams.rollout_length,
                loss_fn=None,
                use_scheduled_sampling=False,
                use_rollout_weighting=False,
            )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)
        self.test_dates.append(torch.concatenate((dates[0][:, -1:], dates[1]), dim=1))
        truth = torch.concatenate((analysis_time, data[1]), dim=1)
        self.test_truth.append(truth)

        tendencies = outs["tendencies"]
        predictions = outs["predictions"]
        predictions = torch.concatenate((analysis_time, predictions), dim=1)
        self.test_predictions.append(predictions)

    def predict_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def _gather(self, predictions, truth, dates, prediction_noo):
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

        if len(predictions_noo):
            predictions_noo = self.all_gather(predictions_noo)
            predictions_noo = predictions_noo.reshape(-1, *predictions_noo.shape[2:])
            predictions_noo = predictions_noo[sort_idx]

        return predictions, truth, dates, predictions_noo

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
            predictions, truth, dates, predictions_noo = self._gather(
                predictions, truth, dates, predictions_noo
            )

        # Only rank 0 writes to avoid duplicate writes
        if self.trainer.is_global_zero:
            _write(predictions, f"{self.test_output_directory}/predictions.pt")

            if len(truth):
                _write(truth, f"{self.test_output_directory}/truth.pt")

            _write(dates, f"{self.test_output_directory}/dates.pt")

            if len(self.test_predictions_noo):
                predictions_noo = torch.concatenate(self.test_predictions_noo)
                _write(
                    predictions_noo, f"{self.test_output_directory}/predictions_noo.pt"
                )

    def on_test_end(self):
        self._write_results_on_test_predict_end()

    def on_predict_end(self):
        self._write_results_on_test_predict_end()

    def configure_optimizers(self):
        decay, no_decay = [], []

        NO_DECAY_KEYS = ["bias", "pos_embed", "norm", "alpha", "gamma", ".A.", ".B."]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in NO_DECAY_KEYS):
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
