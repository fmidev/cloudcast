from lightning.pytorch.cli import SaveConfigCallback
import os
import sys
from argparse import Namespace
from omegaconf import OmegaConf
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def namespace_to_dict_recursive(obj):
    """
    Recursively converts Namespace objects and any other non-standard
    object with a __dict__ (like Path_fsr) into standard dicts, lists, and tuples.
    """
    if isinstance(obj, (list, tuple)):
        # Handle lists and tuples by recurring over their elements
        return type(obj)(namespace_to_dict_recursive(item) for item in obj)

    # Check if the object has a __dict__ but is NOT a basic type
    if hasattr(obj, "__dict__") and not isinstance(
        obj, (str, int, float, bool, type(None))
    ):
        # This handles Namespace, Path_fsr, and any other custom object
        return {k: namespace_to_dict_recursive(v) for k, v in vars(obj).items()}

    # Return everything else (strings, numbers, None, etc.) as is
    return obj


class CustomSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return

        if self.already_saved:
            return

        config_dict = namespace_to_dict_recursive(self.config)
        conf = OmegaConf.create(config_dict)
        rank_zero_info(OmegaConf.to_yaml(conf))

        # Write straight into the run directory rather than deferring to
        # SaveConfigCallback.setup, which resolves trainer.log_dir and asserts
        # it is not None. With a remote MLflow tracking server the logger has
        # no local save_dir, so trainer.log_dir is None and that assert fires.
        run_dir = os.path.join(os.getcwd(), os.environ["CC2_RUN_DIR"])
        config_path = os.path.join(run_dir, "config.yaml")

        if trainer.is_global_zero:
            os.makedirs(run_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            rank_zero_info(f"Saved config to {config_path}")
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved)
