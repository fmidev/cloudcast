import glob
import eccodes as ecc
import sys
import cv2
from datetime import datetime, timedelta
import numpy as np

# Get the list of all GRIB files

grib_files = sorted(glob.glob(f"{sys.argv[1]}*.grib2"))
output_file = sys.argv[2]
print(f"Found {len(grib_files)} GRIB files")

model_data = {"data": [], "time": []}
# Open the first GRIB file to get the number of messages
for grib_file in grib_files:
    with open(grib_file, "rb") as f:
        print(f"Processing {grib_file}")
        datas = []
        times = []
        while True:
            message = ecc.codes_grib_new_from_file(f)
            if message is None:
                break

            dataDate = ecc.codes_get(message, "dataDate")
            dataTime = ecc.codes_get(message, "dataTime")
            step = ecc.codes_get(message, "step")

            ni = ecc.codes_get_long(message, "Ni")
            nj = ecc.codes_get_long(message, "Nj")

            values = ecc.codes_get_values(message).astype(np.float32).reshape(nj, ni)
            #            values = cv2.resize(
            #                values, dsize=(512, 512), interpolation=cv2.INTER_LINEAR
            #            )
            if ecc.codes_get(message, "jScansPositively"):
                values = np.flipud(values)  # image data is +x-y

            datas.append(values)

            ecc.codes_release(message)

            dt = (
                datetime.strptime(f"{dataDate:08d}", "%Y%m%d")
                + timedelta(hours=int(dataTime / 100))
                + timedelta(hours=step)
            )

            times.append(dt)

        datas = np.expand_dims(np.array(datas), axis=-1)
        model_data["data"].append(datas)
        model_data["time"].append(times)

print(f"Saving data to {output_file}")
print(np.asarray(model_data["data"]).shape, np.asarray(model_data["time"]).shape)

ret = {"meps": model_data}

np.savez_compressed(output_file, ret)
