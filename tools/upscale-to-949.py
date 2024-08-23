import numpy as np
import sys
from base.postprocess import downscale
from tqdm import tqdm

outfile = sys.argv[2]
a = np.load(sys.argv[1], allow_pickle=True)
a = a["arr_0"].item()

for model in a.keys():
    if model == "gt":
        continue

    print(model)

    data = a[model]
    times = data["time"]
    datas = data["data"]

    new_data = []
    for data in tqdm(datas):
        new_forecast = []
        for d in data:
            interp = downscale(np.squeeze(d), (1069,949))
            interp = np.expand_dims(interp, axis=-1)
            new_forecast.append(interp)

        new_data.append(np.asarray(new_forecast))

    all_data = {}
    all_data[model] = {"time": times, "data": new_data}

    print(np.asarray(new_data).shape)
    np.savez_compressed(outfile, all_data)

    break
