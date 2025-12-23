#!/usr/bin/env python3
import torch
import os
import randomname
import traceback
import lightning.pytorch as pl
import lightning as L
import json
import time
import random
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
from dataloader.cc2data import cc2DataModule
from common.util import get_next_run_number, get_rank
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.loggers import MLFlowLogger
from common.sc_callback import CustomSaveConfigCallback


def get_coordination_info_identifier() -> str:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        print("SLURM_JOB_ID not set")
        return f"{random.randint(1, 1000):04d}"

    step = os.environ.get("SLURM_STEP_ID")  # may be unset
    return f"{job}" if not step else f"{job}_{step}"


def write_coordination_info(
    coord_file: str, run_name: str, run_number: int, run_dir: str
):
    # Write coordination info to file for other processes
    coord_info = {
        "run_name": run_name,
        "run_number": run_number,
        "run_dir": run_dir,
    }
    with open(coord_file, "w") as f:
        json.dump(coord_info, f)

    print(f"Wrote coordination file {coord_file}")


def setup_run_dir(coord_file: str, ckpt_path: str | None):
    # Generate random name if not already set
    run_name = os.environ.get("CC2_RUN_NAME", randomname.get_name())
    os.environ["CC2_RUN_NAME"] = run_name

    # Get base run directory
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    version = get_next_run_number(run_dir)

    if ckpt_path is not None:
        # If a checkpoint path is provided, use the directory of the checkpoint
        version -= 1

    versioned_dir = os.path.join(run_dir, str(version))
    figures_dir = os.path.join(versioned_dir, "figures")
    checkpoints_dir = os.path.join(versioned_dir, "checkpoints")
    os.makedirs(versioned_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Make these accessible to callbacks
    os.environ["CC2_RUN_NUMBER"] = str(version)
    os.environ["CC2_RUN_DIR"] = versioned_dir

    write_coordination_info(coord_file, run_name, version, versioned_dir)

    return run_name, version, versioned_dir


def initialize_environment(ckpt_path: str | None):
    rank = get_rank()

    coord_file = f"ddp_coordination_info-{get_coordination_info_identifier()}.json"

    if rank == 0:
        setup_run_dir(coord_file, ckpt_path)
    else:
        time.sleep(2)
        max_wait = 20  # seconds
        start_time = time.time()

        while not os.path.exists(coord_file):
            time.sleep(0.2)
            if time.time() - start_time > max_wait:
                raise TimeoutError(
                    f"Coordination file {coord_file} not created after {max_wait} seconds"
                )

        # Read coordination info
        with open(coord_file, "r") as f:
            coord_info = json.load(f)

        os.environ["CC2_RUN_NAME"] = coord_info["run_name"]
        os.environ["CC2_RUN_NUMBER"] = str(coord_info["run_number"])
        os.environ["CC2_RUN_DIR"] = coord_info["run_dir"]

    return coord_file


def setup_mlflow_logger(trainer):
    # setup run name for mlflow logger
    run_name = os.environ["CC2_RUN_NAME"]
    run_number = os.environ["CC2_RUN_NUMBER"]

    assert trainer.loggers is not None and len(
        trainer.loggers
    ), "no loggers defined in configuration"

    for i, logger in enumerate(trainer.loggers):
        if isinstance(logger, L.pytorch.loggers.mlflow.MLFlowLogger):
            new_logger = pl.loggers.MLFlowLogger(
                experiment_name=logger._experiment_name,
                run_name=f"{run_name}/{run_number}",
                save_dir=logger.save_dir,
                log_model=logger._log_model,
                checkpoint_path_prefix=logger._checkpoint_path_prefix,
                tags={"run_name": run_name, "run_number": run_number},
            )
            trainer.loggers[i] = new_logger

            rank_zero_info(f"MLFlowLogger run_name set to: {run_name}")


def _load_stats(ckpt_path):
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("data_statistics") or ckpt.get("hyper_parameters", {}).get(
        "data_statistics"
    )


class cc2trainer(LightningCLI):
    def _inject_statistics(self, stage: str):
        if self.model.use_statistics_from_checkpoint:
            self.datamodule.inject_statistics(_load_stats(self._stage_ckpt(stage)))

    def before_instantiate_classes(self):
        super().before_instantiate_classes()

        if self.subcommand == "fit":
            self.coord_file = initialize_environment(self.config.fit.get("ckpt_path"))

        if self.subcommand == "test":
            ckpt_path = self.config.test.get("ckpt_path")
            assert ckpt_path is not None

            # runs/x-y/1/checkpoints/file.ckpt
            # or
            # /data/runs/era5-big-skip-rollout-1/1/checkpoints/last.ckpt

            parts = ckpt_path.split("/")

            run_dir = "/".join(parts[:-2])
            run_name = parts[-4]
            run_number = parts[-3]

            os.environ["CC2_RUN_NAME"] = run_name
            os.environ["CC2_RUN_NUMBER"] = run_number
            os.environ["CC2_RUN_DIR"] = run_dir

    def before_fit(self):
        self._inject_statistics("fit")

        if self.trainer.global_rank == 0:
            setup_mlflow_logger(self.trainer)

            # Update loggers with version
            for i, logger in enumerate(self.trainer.loggers):
                if isinstance(logger, L.pytorch.loggers.csv_logs.CSVLogger):
                    new_logger = pl.loggers.CSVLogger(
                        save_dir=os.environ["CC2_RUN_DIR"],
                        name="logs" if hasattr(logger, "name") else None,
                        version=None,
                    )
                    self.trainer.loggers[i] = new_logger

            # Update checkpoint callbacks
            for callback in self.trainer.callbacks:
                if callback.__class__.__name__ == "ModelCheckpoint":
                    callback.dirpath = f"{os.environ['CC2_RUN_DIR']}/checkpoints"

            rank_zero_info(f"Run name: {os.environ['CC2_RUN_NAME']}")
            rank_zero_info(f"Version: {os.environ['CC2_RUN_NUMBER']}")
            rank_zero_info(f"Versioned directory: {os.environ['CC2_RUN_DIR']}")

    def _stage_ckpt(self, stage):
        cfg = getattr(self.config, stage, None)
        return getattr(cfg, "ckpt_path", None) if cfg else None

    def before_validate(self):
        self._inject_statistics("validate")

    def before_test(self):
        self._inject_statistics("test")

    def before_predict(self):
        self._inject_statistics("predict")

    def on_train_end(self):
        try:
            os.remove(self.coord_file)
        except FileNotFoundError as e:
            pass


torch.set_float32_matmul_precision("high")

cli = cc2trainer(
    model_class=None,  # read from yaml
    datamodule_class=cc2DataModule,
    save_config_callback=CustomSaveConfigCallback,
    save_config_kwargs={"overwrite": True, "multifile": False},
)
