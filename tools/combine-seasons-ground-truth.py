import numpy as np
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python combine-seasons-ground-truth.py <ground-truth-name> <dir>")
    sys.exit(1)

model = sys.argv[1]
basedir = sys.argv[2]
outputfile = basedir + "/all-seasons/{}-all-seasons.npz".format(model)

if not os.path.exists(os.path.dirname(outputfile)):
    os.makedirs(os.path.dirname(outputfile))


def combine_dicts(dicts):
    combined_dict = {}

    for d in dicts:
        for model, content in d.items():
            if model not in combined_dict:
                combined_dict[model] = {"data": [], "time": []}
            combined_dict[model]["data"].extend(content["data"])
            combined_dict[model]["time"].extend(content["time"])

    return combined_dict


seasons = (
    ["autumn", "winter", "spring", "summer"]
    if "15min" not in model
    else ["spring", "summer"]
)

dicts = []

length = 0
for s in seasons:
    file = f"{basedir}/{s}/{model}-{s}.npz"
    print(f"Loading {file}")
    # Load the data
    data = np.load(file, allow_pickle=True)

    if "arr_0" in data:
        data = data["arr_0"].item()

    dicts.append(data)

    for l in data.keys():
        length += len(data[l]["time"])
        print(length)
        break

# Combine the example dictionaries
combined_dict = combine_dicts(dicts)

# Check that the length of the combined data is correct
for l in combined_dict.keys():
    assert len(combined_dict[l]["time"]) == length
    print(f"Model {l} has {len(combined_dict[l]['time'])} time points")

np.savez(outputfile, combined_dict)

print("Wrote to {}".format(outputfile))
