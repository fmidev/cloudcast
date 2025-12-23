import xarray as xr
import numpy as np
from pyproj import Proj
import argparse


def read_grib(grib_file):
    ds = xr.open_dataset(grib_file, engine="cfgrib", indexpath="")

    # Drop number (ensemble member) coordinate if it exists
    if "number" in ds.coords:
        ds = ds.drop_vars("number")

    return ds


def flatten_forecasts(ds):
    if "step" in ds.dims and len(ds.step) > 1:
        ds = ds.rename({"time": "origintime"})
        ds = ds.assign_coords(time=ds.origintime + ds.step)
        ds = ds.swap_dims({"step": "time"}).drop_vars(["origintime", "step"])
    elif "step" in ds.dims:
        # Single step: just drop it
        ds = ds.isel(step=0)

    # Clean up
    for var in ["step", "valid_time"]:
        if var in ds.coords or var in ds.data_vars:
            ds = ds.drop_vars(var)

    return ds


def setup_projection(ds):
    # Lambert conformal projection from GRIB metadata
    proj = Proj(
        proj="lcc",
        lat_1=63.3,
        lat_2=63.3,
        lat_0=63.3,
        lon_0=15,
        x_0=0,
        y_0=0,
        R=6371229.0,
        units="m",
    )

    # Original grid parameters
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    dx, dy = 2500, 2500  # meters - ORIGINAL resolution
    lon_first, lat_first = 341.879, 72.7617

    x_first, y_first = proj(lon_first, lat_first)

    # Create grid in projection coordinates
    x = x_first + np.arange(nx) * dx
    y = y_first - np.arange(ny) * dy  # Subtract because jScansPositively=0

    xx, yy = np.meshgrid(x, y)

    # Convert back to lat/lon
    lon, lat = proj(xx, yy, inverse=True)

    # Normalize longitude to [-180, 180]
    lon = np.where(lon > 180, lon - 360, lon)

    # Replace coordinates
    ds = ds.drop_vars(["latitude", "longitude"])
    ds["latitude"] = (("y", "x"), lat)
    ds["longitude"] = (("y", "x"), lon)

    print(f"Original grid latitude range: [{lat.min():.2f}, {lat.max():.2f}]")
    print(f"Original grid longitude range: [{lon.min():.2f}, {lon.max():.2f}]")

    return ds


def flip_horizontal(ds):
    ds = ds.isel(y=slice(None, None, -1))
    return ds


def upscale(ds, target_y=535, target_x=475):
    original_shape = (ds.sizes["y"], ds.sizes["x"])

    # Interpolate both data and lat/lon coordinates
    new_y = np.linspace(0, ds.sizes["y"] - 1, target_y)
    new_x = np.linspace(0, ds.sizes["x"] - 1, target_x)

    ds_interp = ds.interp(y=new_y, x=new_x, method="linear")

    # Rebuild dataset with clean integer coordinates
    data_vars = {}
    coords = {}

    # Copy data variables (exclude coordinates)
    for var in ds_interp.data_vars:
        if var not in ["latitude", "longitude", "x", "y"]:
            data_vars[var] = (
                ds_interp[var].dims,
                ds_interp[var].values,
                ds_interp[var].attrs,
            )

    # Add clean integer x/y coordinates
    coords["x"] = np.arange(target_x)
    coords["y"] = np.arange(target_y)

    # Add interpolated lat/lon as 2D coordinates
    coords["latitude"] = (("y", "x"), ds_interp.latitude.values)
    coords["longitude"] = (("y", "x"), ds_interp.longitude.values)

    # Copy scalar/1D coordinates (like time)
    for coord in ds_interp.coords:
        if coord not in ["x", "y", "latitude", "longitude"]:
            coords[coord] = ds_interp[coord]

    ds_new = xr.Dataset(data_vars, coords=coords)

    print(f"Upscaled from {original_shape} to ({target_y}, {target_x})")
    print(
        f"Final latitude range: [{ds_new.latitude.values.min():.2f}, {ds_new.latitude.values.max():.2f}]"
    )
    print(
        f"Final longitude range: [{ds_new.longitude.values.min():.2f}, {ds_new.longitude.values.max():.2f}]"
    )

    return ds_new


def flatten_levels_and_clean_attrs(ds):
    data_vars = {}
    coords = {}

    # Copy coordinates (excluding isobaricInhPa)
    for coord in ds.coords:
        if coord not in ["isobaricInhPa"]:
            coords[coord] = ds[coord]

    # Process each variable
    for var in ds.data_vars:
        if "isobaricInhPa" in ds[var].dims:
            # Flatten: create separate variable for each level
            for level in ds.isobaricInhPa.values:
                level_int = int(level)
                new_var_name = f"{var}_{level_int}"
                data = ds[var].sel(isobaricInhPa=level).drop_vars("isobaricInhPa")

                # Clean GRIB attributes and remove 'coordinates' attribute
                attrs = {
                    k: v
                    for k, v in data.attrs.items()
                    if not k.startswith("GRIB") and k != "coordinates"
                }
                data_vars[new_var_name] = (data.dims, data.values, attrs)
        else:
            # No level dimension, just copy
            attrs = {
                k: v
                for k, v in ds[var].attrs.items()
                if not k.startswith("GRIB") and k != "coordinates"
            }
            data_vars[var] = (ds[var].dims, ds[var].values, attrs)

    ds_new = xr.Dataset(data_vars, coords=coords)

    print(f"Flattened {len(ds.isobaricInhPa)} levels into separate variables")
    return ds_new


def clean_grib_attrs(ds):
    for var in ds.data_vars:
        ds[var].attrs = {
            k: v
            for k, v in ds[var].attrs.items()
            if not k.startswith("GRIB") and k != "coordinates"
        }
    return ds


def pad_time(ds, target_timesteps):
    current_timesteps = ds.sizes["time"]

    if current_timesteps >= target_timesteps:
        print(
            f"Time dimension already has {current_timesteps} timesteps, no padding needed"
        )
        return ds

    # Get time coordinate and calculate hourly increments
    times = ds.time.values
    time_delta = np.timedelta64(1, "h")  # 1 hour

    # Create new time coordinates
    new_times = [
        times[-1] + time_delta * (i + 1)
        for i in range(target_timesteps - current_timesteps)
    ]
    all_times = np.concatenate([times, new_times])

    # Create padded dataset
    data_vars = {}
    coords = {}

    # Pad each data variable
    for var in ds.data_vars:
        if "time" in ds[var].dims:
            # Create padded array with NaN
            pad_shape = list(ds[var].shape)
            time_axis = ds[var].dims.index("time")
            pad_shape[time_axis] = target_timesteps - current_timesteps

            pad_array = np.full(pad_shape, np.nan, dtype=ds[var].dtype)
            padded_data = np.concatenate([ds[var].values, pad_array], axis=time_axis)

            data_vars[var] = (ds[var].dims, padded_data, ds[var].attrs)
        else:
            # No time dimension, just copy
            data_vars[var] = (ds[var].dims, ds[var].values, ds[var].attrs)

    # Update time coordinate
    coords["time"] = all_times

    # Copy other coordinates
    for coord in ds.coords:
        if coord != "time":
            coords[coord] = ds[coord]

    ds_padded = xr.Dataset(data_vars, coords=coords)

    print(
        f"Padded time dimension from {current_timesteps} to {target_timesteps} timesteps"
    )
    return ds_padded


def write_netcdf(ds, netcdf_file):
    if "time" in ds.coords:
        ds["time"].attrs["standard_name"] = "time"
        ds["time"].attrs["long_name"] = "time"

    ds.to_netcdf(netcdf_file)
    print(f"Written to {netcdf_file}")


def process_meps(grib_file, netcdf_file, prediction_length):
    print("Processing MEPS data...")
    ds = read_grib(grib_file)
    ds = flatten_forecasts(ds)
    ds = flip_horizontal(ds)
    ds = setup_projection(ds)
    ds = upscale(ds, target_y=535, target_x=475)
    ds = flatten_levels_and_clean_attrs(ds)
    write_netcdf(ds, netcdf_file)


def process_nwcsaf(grib_file, netcdf_file, prediction_length):
    print("Processing NWCSAF data...")
    ds = read_grib(grib_file)
    ds = flatten_forecasts(ds)
    ds = flip_horizontal(ds)
    ds = setup_projection(ds)
    ds = upscale(ds, target_y=535, target_x=475)
    ds = clean_grib_attrs(ds)
    ds = pad_time(ds, 2 + prediction_length)
    write_netcdf(ds, netcdf_file)


def main():
    parser = argparse.ArgumentParser(description="Convert GRIB2 to NetCDF")
    parser.add_argument("input", help="Input GRIB file")
    parser.add_argument("output", help="Output NetCDF file")
    parser.add_argument(
        "--producer",
        choices=["meps", "nwcsaf"],
        required=True,
        help="Data producer (meps or nwcsaf)",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=8,
        help="Target prediction length for padding (default: 8)",
    )

    args = parser.parse_args()

    if args.producer == "meps":
        process_meps(args.input, args.output, args.prediction_length)
    elif args.producer == "nwcsaf":
        process_nwcsaf(args.input, args.output, args.prediction_length)


if __name__ == "__main__":
    main()
