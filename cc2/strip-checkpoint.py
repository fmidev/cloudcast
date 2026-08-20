#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch

# Training-only state: large and useless for inference/deployment.
DROP_KEYS = ("optimizer_states", "lr_schedulers", "loops", "callbacks")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Strip a Lightning checkpoint for deployment: drop optimizer/"
            "scheduler/loop/callback state, keep model weights, hyper_parameters "
            "and data_statistics."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file")
    parser.add_argument(
        "--out", required=True, help="Path to output stripped .ckpt file"
    )

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise SystemExit(f"{ckpt_path} does not contain 'state_dict'")

    if "data_statistics" not in ckpt:
        raise SystemExit(
            f"{ckpt_path} has no data_statistics: a deploy checkpoint without "
            "them breaks use_statistics_from_checkpoint at inference time."
        )

    stripped = {k: v for k, v in ckpt.items() if k not in DROP_KEYS}
    dropped = [k for k in ckpt if k in DROP_KEYS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, out_path)

    in_mb = os.path.getsize(ckpt_path) / 1e6
    out_mb = os.path.getsize(out_path) / 1e6
    print(f"Read checkpoint: {ckpt_path} ({in_mb:.0f} MB)")
    print(f"Dropped keys: {dropped}")
    print(f"Kept keys: {list(stripped.keys())}")
    print(f"Saved stripped checkpoint to: {out_path} ({out_mb:.0f} MB)")


if __name__ == "__main__":
    main()
