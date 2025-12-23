import argparse
import numpy as np
import xarray as xr
import torch
from grib import save_grib

def load_tensors(tensor_path, dates_path):
    data = torch.load(tensor_path, map_location="cpu", weights_only=True).numpy()
    dates = torch.load(dates_path, map_location="cpu", weights_only=True).numpy()

    B, T, C, H, W = data.shape
    print(f"Loaded tensor: B={B}, T={T}, C={C}, H={H}, W={W}")

    assert C == 1, "Channel size must be 1"

    # Extract first channel
    data = data[:, :, 0]

    return data, dates


def downscale(data, target_height=1069, target_width=949):
    B, T, H, W = data.shape
    print(f"Downscaling from ({H}, {W}) to ({target_height}, {target_width})")

    # Interpolate coordinates once
    new_y = np.linspace(0, H - 1, target_height)
    new_x = np.linspace(0, W - 1, target_width)

    downscaled_batches = []

    for b in range(B):
        ds = xr.DataArray(
            data[b],
            dims=["time", "y", "x"],
            coords={"time": np.arange(T), "y": np.arange(H), "x": np.arange(W)},
        )
        downscaled = ds.interp(y=new_y, x=new_x, method="linear")
        downscaled_batches.append(downscaled.values)

    return np.stack(downscaled_batches, axis=0)


def flip_latitude(data):
    print("Flipping latitude axis")
    return data[:, :, ::-1, :]


def main():
    parser = argparse.ArgumentParser(
        description="Convert model output tensor back to GRIB format"
    )
    parser.add_argument(
        "--output-tensor", required=True, help="Model output tensor file (.pt)"
    )
    parser.add_argument("--dates-tensor", required=True, help="Dates tensor file (.pt)")
    parser.add_argument("--output-grib", required=True, help="Output GRIB file path")
    parser.add_argument("--grib-options", default=None, help="Additional GRIB options")

    args = parser.parse_args()

    # Pipeline
    data, dates = load_tensors(args.output_tensor, args.dates_tensor)
    data = downscale(data, target_height=1069, target_width=949)
    data = flip_latitude(data)
    save_grib(data, dates, args.output_grib, args.grib_options)


if __name__ == "__main__":
    main()
