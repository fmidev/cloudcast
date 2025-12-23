set -uex

CC2_OUTPUT_PATH="${CC2_OUTPUT_PATH:-$dir/output}"

python3 downscale-to-meps-domain.py --output-tensor $CC2_OUTPUT_PATH/predictions.pt --dates-tensor $CC2_OUTPUT_PATH/dates.pt --output-grib $CC2_OUTPUT_PATH/predictions.grib2
