import numpy as np
import sys
from datetime import datetime

outfile = sys.argv[2]
data = np.load(sys.argv[1], allow_pickle=True)

all_data = {}

times = data["arr_1"]
times = [datetime.strptime(time, "%Y%m%dT%H%M%S") for time in times]
datas = data["arr_0"]

new_time = []
new_data = []
for time, data in zip(times, datas):

    if time.minute != 0:
        continue

    print(time)

    new_time.append(time)
    new_data.append(data)

np.savez_compressed(outfile, {"gt": {"time": new_time, "data": new_data}})
