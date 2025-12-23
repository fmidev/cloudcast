from cc2data import cc2DataModule
import matplotlib.pyplot as plt
import torch as t
import zarr
import os
import sys
import platform
import math


def print_env():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {t.__version__}")
    print(f"Zarr version: {zarr.__version__}")
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Filesystem info:")

    try:
        import subprocess

        print(subprocess.check_output(["df", "-T", "."]).decode())
    except:
        print("Couldn't get filesystem type")

    # Also check ulimit
    try:
        import resource

        print(f"Max open files: {resource.getrlimit(resource.RLIMIT_NOFILE)}")
    except:
        print("Couldn't get ulimit info")


print_env()

zarr_path = (
    "/anemoi-data/cerra-475x535-1994-1998-v3.zarr"
    if len(sys.argv) == 1
    else sys.argv[1]
)

forcings = [
    "insolation",
    "cos_julian_day",
    "sin_julian_day",
    "cos_local_time",
    "sin_local_time",
    "t_1000",
    "t_500",
    "t_700",
    "t_850",
    "t_925",
    "u_1000",
    "u_500",
    "u_700",
    "u_850",
    "u_925",
    "v_1000",
    "v_500",
    "v_700",
    "v_850",
    "v_925",
    "z_1000",
    "z_500",
    "z_700",
    "z_850",
    "z_925",
    "r_1000",
    "r_500",
    "r_700",
    "r_850",
    "r_925",
]

cc2Data = cc2DataModule(
    data_path=zarr_path,
    input_resolution=[475, 535],
    prognostic_params=["tcc"],
    forcing_params=forcings,
    batch_size=1,
)
cc2Data.setup("fit")

df = cc2Data.train_dataloader()

for i, batch in enumerate(df):
    data, forcing = batch
    x, y = data
    print("Batch", i, "X shape", x.shape, "Y shape", y.shape)

    x_f = x[0]
    y_f = y[0]

    print(
        "X min={:.3f} mean={:.3f} max={:.3f}".format(
            t.min(x_f), t.mean(x_f), t.max(x_f)
        )
    )
    print(
        "Y min={:.3f} mean={:.3f} max={:.3f}".format(
            t.min(y_f), t.mean(y_f), t.max(y_f)
        )
    )

    print(
        "Forcing min={:.3f} mean={:.3f} max={:.3f}".format(
            t.min(forcing), t.mean(forcing), t.max(forcing)
        )
    )

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    ax[0].imshow(x_f[0].squeeze())
    ax[1].imshow(x_f[1].squeeze())
    ax[2].imshow(y_f.squeeze())
    plt.savefig("figures/test-dataloader-prognostic-timeseries.png")
    print("Saved figure: figures/test-dataloader-prognostic-timeseries.png")

    f_f = forcing[0]

    prog = t.cat((x_f, y_f), dim=0)

    var = t.cat((prog[0], f_f[0]))
    n = var.shape[0]

    # determine grid size (nearly square)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    labels = forcings + ["tcc-1", "tcc", "tcc+1"]
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i in range(n):
        v = var[i]
        r, c = divmod(i, cols)
        ax = axes[r][c]

        im = ax.imshow(v)
        fig.colorbar(im, ax=ax)  # , label=str(var))
        ax.set_title(labels[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # turn off any empty subplots
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig("figures/test-dataloader-variables.png")

    print("Saved figure: figures/test-dataloader-variables.png")
    break
