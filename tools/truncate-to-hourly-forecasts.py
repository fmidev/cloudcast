import numpy as np
import sys
from tqdm import tqdm

outfile = sys.argv[2]
a = np.load(sys.argv[1], allow_pickle=True)
a = a["arr_0"].item()

all_data = {}

for model in a.keys():
    if model == "gt":
        gt_time = []
        gt_data = []

        for time, data in zip(a["gt"]["time"], a["gt"]["data"]):
            gt_time.append(time)
            gt_data.append(data)

        all_data["gt"] = {"time": gt_time, "data": gt_data}
        continue

    print(model)

    data = a[model]
    times = data["time"]
    datas = data["data"]

    new_time = []
    new_data = []
    for time, data in tqdm(zip(times, datas)):
        assert len(time) == 21
        assert data.shape[0] == 21

        analysis_time = time[0]
        if analysis_time.minute != 0:
            continue

        print(analysis_time)

        new_time.append(time)
        new_data.append(data)

    all_data[model] = {"time": new_time, "data": new_data}


np.savez_compressed(outfile, all_data)
