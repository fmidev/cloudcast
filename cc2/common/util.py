import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import os
import torch
import time
import importlib
from glob import glob


def strip_prefix(state_dict: dict, prefix: str = "model."):
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def find_latest_checkpoint_path(checkpoint_directory):
    assert checkpoint_directory is not None, "checkpoint_directory is 'None'"
    try:
        best = f"{checkpoint_directory}/checkpoints/best.ckpt"
        if os.path.exists(best):
            return best
        # Find latest checkpoint
        checkpoints = glob(f"{checkpoint_directory}/checkpoints/epoch*.ckpt")
        assert (
            checkpoints
        ), f"No model checkpoints found in directory {checkpoint_directory}/checkpoints, cwd={os.getcwd()}"
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        return latest_ckpt
    except ValueError as e:
        print("Model checkpoint file not found from path: {}/models".format(file_path))
        raise e


def get_next_run_number(base_dir):
    rank = get_rank()
    next_run_file = f"next_run.txt"

    assert rank == 0, f"rank is {rank}, not 0"

    # Only rank 0 determines the next run number
    if not os.path.exists(base_dir):
        next_num = 1
    else:
        subdirs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
        ]
        next_num = max([int(d) for d in subdirs], default=0) + 1

    return next_num


def get_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None

    subdirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    if not subdirs:
        return None
    latest = max([int(d) for d in subdirs])
    return os.path.join(base_dir, str(latest))


def get_rank():
    # 1) If DDP/FSDP has already inited, use it
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    # 2) torchrun / Lightning will set RANK and LOCAL_RANK
    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    # 3) SLURM’s local‐id (if you still need it for sbatch)
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])

    # 4) last‐resort
    return 0


def moving_average(arr, window_size):
    """
    Calculate the running mean of a 1D array using a sliding window.

    Parameters:
    - arr (torch.Tensor): 1D input array.
    - window_size (int): The size of the sliding window.

    Returns:
    - torch.Tensor: The running mean array.
    """
    # Ensure input is a 1D tensor
    arr = arr.reshape(1, 1, -1)

    if window_size >= arr.shape[-1]:
        return torch.full((arr.shape[-1],), float("nan"))

    # Create a uniform kernel
    kernel = torch.ones(1, 1, window_size) / window_size

    # Apply 1D convolution
    running_mean = torch.nn.functional.conv1d(
        arr, kernel, padding=0, stride=1
    ).squeeze()

    nan_padding = torch.full((window_size - 1,), float("nan"))

    result = torch.cat((nan_padding, running_mean))

    return result


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_directory_structure(base_directory):
    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(f"{base_directory}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_directory}/logs", exist_ok=True)
    os.makedirs(f"{base_directory}/figures", exist_ok=True)
