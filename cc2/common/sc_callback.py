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
        path = os.getcwd()
        self.config_filename = f"{path}/{os.environ['CC2_RUN_DIR']}/config.yaml"
        super().setup(trainer, pl_module, stage)
