set -uex

CC2_DATA_PATH="${CC2_DATA_PATH:-data}"
CC2_ETC_PATH="${CC2_ETC_PATH:-etc}"

python3 python/prepare-data.py --producer nwcsaf $CC2_DATA_PATH/nwcsaf-surface.grib2 $CC2_DATA_PATH/nwcsaf-surface.nc
python3 python/prepare-data.py --producer meps $CC2_DATA_PATH/meps-pressure.grib2 $CC2_DATA_PATH/meps-pressure.nc

export atime_start=$(date -ud "${ANALYSIS_TIME}z 1 hours ago" +"%Y-%m-%d %H:00:00")
export atime_end=$(date -ud "${ANALYSIS_TIME}z +$MAX_HOURS hours" +"%Y-%m-%d %H:00:00")

rm -rf $CC2_DATA_PATH/meps-nwcsaf.zarr

yq '.dates.start = strenv(atime_start) | .dates.end = strenv(atime_end) | .statistics.end = strenv(atime_start)' $CC2_ETC_PATH/cc2.yaml | \
	tee /dev/fd/2 | \
        anemoi-datasets create - $CC2_DATA_PATH/meps-nwcsaf.zarr

anemoi-datasets inspect $CC2_DATA_PATH/meps-nwcsaf.zarr
