# Running inference with Cloudcast v2

This guide provides instructions on how to run inference using the Cloudcast v2 model. 

All scripts assume two environment variables are set:
* ANALYSIS_TIME: analysis time in format YYYY-MM-DD HH:MM:SS, this corresponds to the second history step in the data files
* MAX_TIME: maximum forecast time in hours (e.g., 8 for 8-hour forecast).

The scripts also assume they are ran from the inference directory of the repository.

The basic workflow includes three main steps:
1. Prepare initial conditions
2. Run inference
3. Post-process results

## 1. Prepare initial conditions

To run inference we need initial conditions from both MEPS and NWCSAF.

### 1.1 MEPS data

From MEPS, we need:
* variables: u, v, t, z, r
* pressure levels: 1000, 925, 850, 700, 500 hPa
* lsm, orography (static fields)

This guide assumes the data is in GRIB format. MEPS data can be download from MET Norway's Thredds archive.

The following is assumed from MEPS data:

* data is in GRIB format
* the variables given above are available
* data contains two history steps + 8 forecast steps i.e., 10 time steps in total
  * it does not matter whether the data is forecast or analysis data, as long as there are 10 consecutive time steps
* data is on the native MEPS grid (2.5 km resolution, 1069x949 pixels)
* first point is bottom-left corner

### 1.2 NWCSAF data

From NWCSAF, we need:
* variable: effective cloudiness (0 to 1)

This guide assumes the data is in GRIB format. NWCSAF data is produced typically by national Met services.

The following is assumed from NWCSAF data:

* data is in GRIB format
* the variables given above are available
* data contains two history steps which correspond to the two history steps of MEPS data
* data is on the native MEPS grid (2.5 km resolution, 1069x949 pixels)
* first point is bottom-left corner

### 1.3 Prepare data files

Place the MEPS and NWCSAF data files in the `data/` directory in files `meps-pressure.grib2` and `nwcsaf-surface.grib2`, respectively. Also copy the static fields file `meps-const-v4.zarr` to `data/` directory (anonymous access available at s3://cloudcast-v2/const/meps-const-v4.zarr, endpoint_url=https://lake.fmi.fi).

Run conversion script:

```python
sh bin/convert.sh
```

The script will:
* upscale the data to 5 km resolution
* flatten the MEPS time dimension (remove the origintime+step structure)
* flip the data so that first point is top-left corner
* convert to NetCDF format
  * NetCDF is needed because eccodes does not calculate coordinates correctly for lcc +x-y grids in GRIB 
* create anemoi-dataset archive
* store the results in `data/` directory

Output directory location can be controlled with environment variable `CC2_DATA_DIR`, default is `data/`.

## 2. Run inference

To run inference, use the following command:

```python
sh run.sh <path_to_config> <path_to_checkpoint>
```

Result is two torch tensor written in `output/` directory:
* `predictions.pt`: the forecasted cloud fields
* `dates.pt`: the dates associated for each prediction

Output directory location can be controlled with environment variable `CC2_OUTPUT_DIR`, default is `output/`.

## 3. Post-processing

Post-processing includes the following steps:
* downscaling the predictions back to 2.5 km resolution and MEPS grid
* flipping the data so that first point is bottom-left corner
* writing the results as GRIB

Run post-processing script:

```python
sh bin/postprocess.sh
```

The resulting GRIB file will be written to `output/predictions.grib2`.

Output directory location can be controlled with environment variable `CC2_OUTPUT_DIR`, default is `output/`.
