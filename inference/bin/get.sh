set -exu

echo $ANALYSIS_TIME

CC2_DATA_PATH="${CC2_DATA_PATH:-data}"
CC2_ETC_PATH="${CC2_ETC_PATH:-etc}"

# Fetch NWCSAF

rm -f $CC2_DATA_PATH/meps-*.grib2 $CC2_DATA_PATH/nwcsaf-*.grib2 $CC2_DATA_PATH/nwcsaf-*.nc $CC2_DATA_PATH/meps-*.nc

prev_cc2_atime=$(date -ud "${ANALYSIS_TIME}z - 1 hour" +"%Y-%m-%d %H:00:00")

jq --arg atimes "${prev_cc2_atime},${ANALYSIS_TIME}" \
   '.origintimes = $atimes' \
   $CC2_ETC_PATH/cc2-nwcsaf.json | \
   tee /dev/fd/2 | \
   himan -f -

# Fetch MEPS

h=$(date -ud "${ANALYSIS_TIME}z" +"%H")
h=$(echo $h | sed 's/^0*//') # Remove leading zeros

if [ $(( $h % 3 )) -eq 0 ]; then
  h=3
else
  h=$(( $h % 3 + 3 ))
fi

latest_meps=$(date -ud "${ANALYSIS_TIME}z - $h hours" +"%Y-%m-%d %H:00:00")

MAX_FORECAST_HOUR=$((MAX_HOURS + 1 + 2)) # 8h including 0 step and 2 history steps
hours=""
if [ $h -eq 2 ]; then
  MAX_FORECAST_HOUR=$(( MAX_FORECAST_HOUR + 1 ))
  hours="$(seq -s ',' 1 $MAX_FORECAST_HOUR)"
elif [ $h -eq 3 ]; then
  MAX_FORECAST_HOUR=$(( MAX_FORECAST_HOUR + 2 ))
  hours="$(seq -s ',' 2 $MAX_FORECAST_HOUR)"
elif [ $h -eq 4 ]; then
  MAX_FORECAST_HOUR=$(( MAX_FORECAST_HOUR + 3 ))
  hours="$(seq -s ',' 3 $MAX_FORECAST_HOUR)"
elif [ $h -eq 5 ]; then
  MAX_FORECAST_HOUR=$(( MAX_FORECAST_HOUR + 4 ))
  hours="$(seq -s ',' 4 $MAX_FORECAST_HOUR)"
else
  echo "Invalid h value: $h"
  exit 1
fi

jq --arg latest_meps "${latest_meps}" \
   --arg hours "${hours}" \
   '.origintime = $latest_meps | .hours = $hours' \
   $CC2_ETC_PATH/cc2-meps.json | \
   tee /dev/fd/2 | \
   himan -f -
