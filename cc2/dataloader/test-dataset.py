from cc2data import AnemoiDataset
import matplotlib.pyplot as plt
import torch as t
import zarr
import os
import sys
import platform

zarr_path = "../data/nwcsaf-128x128-hourly-anemoi.zarr"
ds = AnemoiDataset(zarr_path, 2 + 1, (128, 128))

print(ds)

for batch in ds:
    data = batch
    print(data.shape)

    data = data.permute(1, 0, 2, 3).reshape(data.shape[1], -1)
    print(
        "min={} mean={} max={}".format(t.min(data, 0), t.mean(data, 0), t.max(data, 0))
    )
    break
