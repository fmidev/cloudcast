import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from datetime import timedelta
from osgeo import osr
from base.s3utils import *
from io import BytesIO
import os
import itertools
from matplotlib.colors import LinearSegmentedColormap

FIGURE = 0


def figure():
    global FIGURE
    FIGURE += 1
    return FIGURE


def savefig(plot_dir):
    fname = "{}/figure{:02d}.png".format(plot_dir, FIGURE)

    if fname[0:5] == "s3://":
        img_data = BytesIO()
        plt.savefig(img_data)
        img_data.seek(0)
        write_to_s3(fname, img_data)
    else:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(fname)

    print(f"Saved {fname}")


def reduce_label(label):
    return (
        label.replace("dt=False-", "")
        .replace("topo=False-", "")
        .replace("terrain=False-", "")
        .replace("sun=False-", "")
        .replace("oh=False-", "")
        .replace("True", "T")
        .replace("False", "F")
        .replace("binary_crossentropy", "bc")
        .replace("MeanSquaredError", "MSE")
        .replace("MeanAbsoluteError", "MAE")
        .replace("-img_size=128x128", "-128")
        .replace("-img_size=256x256", "-256")
        .replace("-img_size=512x512", "-512")
        .replace("-img_size=768x768", "-768")
        .replace("-lc=12", "")
        .replace("-hist=4", "")
    )


def latlonraster(img_size):
    # Data axis to CRS axis mapping: 1,2
    # Origin = (-1072595.173187759937719,9675899.727970723062754)
    # Pixel Size = (18535.155999999999040,-20878.905999999999040)
    # Corner Coordinates:
    # Upper Left  (-1072595.173, 9675899.728) ( 18d 7'31.78"W, 72d37'10.55"N)
    # Lower Left  (-1072595.173, 7003399.760) (  0d11'14.17"E, 50d12'42.21"N)
    # Upper Right ( 1299904.795, 9675899.728) ( 53d39'41.09"E, 71d33' 5.48"N)
    # Lower Right ( 1299904.795, 7003399.760) ( 32d48'21.68"E, 49d42'34.98"N)
    # Center      (  113654.811, 8339649.744) ( 17d15'33.72"E, 63d12'24.20"N)
    # Band 1 Block=128x1 Type=Float64, ColorInterp=Undefined

    src = osr.SpatialReference()
    tgt = osr.SpatialReference()
    src.ImportFromProj4(
        "+proj=lcc +lat_0=0 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    tgt.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(src, tgt)

    if img_size == (128, 128):
        x = np.linspace(-1072595.173, 1299904.795, 128)
        y = np.linspace(9675899.728, 7003399.760, 128)
    elif img_size == (256, 256):
        x = np.linspace(-1067961.384, 1304538.584, 256)
        y = np.linspace(9681119.454, 7008619.486, 256)
    elif img_size == (384, 384):
        x = np.linspace(-1066416.788, 1306083.052, 384)
        y = np.linspace(9682859.235, 7010359.395, 384)
    elif img_size == (512, 512):
        x = np.linspace(-1065644.490, 1306855.478, 512)
        y = np.linspace(9683729.573, 7011229.349, 512)
    elif img_size == (768, 768):
        x = np.linspace(-1065644.490, 1306855.478, 768)
        y = np.linspace(9683729.573, 7011229.349, 768)
    else:
        raise Exception("Unsupported img_size for 2d plotting: {}".format(img_size))

    lon = []
    lat = []

    for y_ in y:
        lon.append([])
        lat.append([])
        for x_ in x:
            lat_, lon_, _ = transform.TransformPoint(x_, y_)
            lon[-1].append(lon_)
            lat[-1].append(lat_)

    return np.asarray(lon), np.asarray(lat)


def plot_on_map(data, title=None, plot_dir=None):
    plt.close("all")
    plt.figure(figure(), figsize=(10, 8))
    m = Basemap(
        llcrnrlon=-0.3,
        llcrnrlat=49.6,
        urcrnrlon=57.3,
        urcrnrlat=71.8,
        ellps="WGS84",
        resolution="l",
        area_thresh=1000.0,
        projection="lcc",
        lat_1=63.0,
        lat_2=63,
        lat_0=63,
        lon_0=15.0,
    )

    colors = [(1, 1, 1), (1, 0, 0)]  # White to red
    n_bins = 100  # Number of bins in the colormap
    cmap_name = "white_to_red"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    lons, lats = latlonraster(data.shape)

    x, y = m(lons, lats)
    cs = m.pcolormesh(x, y, data, shading="auto", vmin=0, vmax=0.7, cmap=cm)

    m.drawcoastlines()
    m.drawmapboundary()
    m.drawparallels(np.arange(-90.0, 120.0, 30.0), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180.0, 180.0, 15.0), labels=[0, 0, 0, 1])

    plt.colorbar(cs, orientation="vertical", shrink=0.5)
    plt.title(title)

    if plot_dir is not None:
        savefig(plot_dir)


def plot_normal(x, y, y2, labels, title=None, xlabels=None, plot_dir=None):
    plt.close("all")
    assert len(labels) == len(y)
    fig = plt.figure(figure(), figsize=(12, 7))
    ax1 = plt.axes()
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    xreal = np.asarray(range(len(x)))

    labels = list(map(lambda x: reduce_label(x), labels))

    color = "tab:grey"

    ax1.set_xlabel("time")
    ax1.set_ylabel("count")
    ax1.scatter(x, y2, color=color, marker="x", s=3, label="counts")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    for i, y_ in enumerate(y):
        y_ = np.asarray(y_)
        ax2.plot(x, y_, label=labels[i], linestyle="-", marker="o", markersize=2)
        ax2.set_ylabel("mean absolute error")

    plt.title(title)
    plt.gcf().autofmt_xdate()
    ax2.legend()

    if plot_dir is not None:
        savefig(plot_dir)


def plot_bargraph(data, labels, title=None, xvalues=None, ylabel=None, plot_dir=None):
    plt.close("all")
    assert len(data) == len(labels)
    labels = list(map(lambda x: reduce_label(x), labels))

    for i, _data in enumerate(data):
        x = _data[1]
        y = _data[0]
        x = x[:-1]
        fig = plt.figure(figure(), figsize=(10, 6))
        ax = plt.axes()
        ax.set_xlabel("bins")
        ax.set_ylabel(ylabel)

        label = labels[i]
        # plt.stairs(x, y, label=label)
        title = "histogram for {}".format(labels[i])
        plt.title(title)

        plt.hist(x, 50, weights=y)
    #   ax.set_xticks(xreal)
    #   ax.set_xticklabels(xlabels)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if plot_dir is not None:
        savefig(plot_dir)


def plot_linegraph(
    data,
    labels,
    title=None,
    xvalues=None,
    ylabel=None,
    plot_dir=None,
    start_from_zero=False,
    add_mean_value_to_label=False,
    full_hours_only=False,
    errors=None,
):
    plt.close("all")
    assert len(data) == len(labels)
    assert errors is None or len(errors) == len(data)

    fig = plt.figure(figure(), figsize=(12, 7))
    ax = plt.axes()
    ax.set_xlabel("leadtime")
    ax.set_ylabel(ylabel)
    step = timedelta(minutes=60) if full_hours_only else timedelta(minutes=15)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    xreal = np.asarray(range(len(data[0])))

    offset = 0 if start_from_zero else 1

    xlabels = list(map(lambda x: step * x, range(offset, offset + len(data[0]))))
    xlabels = list(map(lambda x: "{}".format(int(x.total_seconds() / 60)), xlabels))

    labels = list(map(lambda x: reduce_label(x), labels))

    if errors is None:
        for i, mae in enumerate(data):
            mae = np.asarray(mae)
            x = xreal[np.isfinite(mae)]
            y = mae[np.isfinite(mae)]

            label = labels[i]
            if add_mean_value_to_label:
                label = "{} mean: {:.3f}".format(label, np.mean(mae))
            ax.plot(x, y, label=label, linestyle="-", marker="o")
    else:
        for i, mae in enumerate(data):
            mae = np.asarray(mae)
            x = xreal[np.isfinite(mae)]
            y = mae[np.isfinite(mae)]

            label = labels[i]
            if add_mean_value_to_label:
                label = "{} mean: {:.3f}".format(label, np.mean(mae))
            ax.errorbar(x, y, yerr=errors[i], label=label, linestyle="-", marker="o")

    ax.set_xticks(xreal)
    ax.set_xticklabels(xlabels)
    plt.title(title)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if plot_dir is not None:
        savefig(plot_dir)


def plot_stamps(
    datas,
    labels,
    title=None,
    initial_data=None,
    start_from_zero=False,
    full_hours_only=False,
    plot_dir=None,
):
    assert len(datas) == len(labels)

    for i in range(1,len(datas)):
        assert datas[i].shape == datas[0].shape

    plt.close("all")
    nrows = len(datas)
    ncols = datas[0].shape[0]
    if initial_data is not None:
        nrows += 1
        labels = ["initial"] + labels

    fig, bigaxes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=((ncols * 2.5), nrows * 3.5),
        constrained_layout=False,
        squeeze=False,
        num=figure(),
    )
    fig.suptitle(title)
    for i, bigax in enumerate(bigaxes.flatten(), start=0):
        bigax.set_title(labels[i])
        bigax.tick_params(
            labelcolor=(1.0, 1.0, 1.0, 0.0),
            top="off",
            bottom="off",
            left="off",
            right="off",
        )
        bigax._frameon = False
        bigax.axis("off")

    num = 1
    write_time = nrows - 1

    if initial_data is not None:
        ax = fig.add_subplot(nrows, ncols, num)
        ax.imshow(np.squeeze(initial_data), cmap="gray_r")
        ax.axis("off")
        num = ncols + 1
        write_time = nrows - 2

    offset = 0 if start_from_zero else 1
    factor = 60 if full_hours_only else 15

    for i in range(len(datas)):
        for j in range(datas[i].shape[0]):
            ax = fig.add_subplot(nrows, ncols, num)
            num += 1
            ax.imshow(np.squeeze(datas[i][j]), cmap="gray_r")
            ax.axis("off")
            if i == write_time:
                ax.set_title(f"{(j+offset)*factor}", y=0, pad=-25)

    fig.set_facecolor("w")

    if plot_dir is not None:
        savefig(plot_dir)


def plot_histogram(datas, labels, plot_dir=None):
    assert len(datas) == len(labels)
    n_bins = 50
    plt.close("all")
    fig, axs = plt.subplots(
        1, len(datas), sharey=True, tight_layout=False, num=figure()
    )
    fig.set_size_inches(12, 8)

    if len(datas) == 1:
        axs.hist(np.asarray(datas[0]).flatten(), bins=n_bins, density=True)
        axs.set_title(reduce_label(labels[0]))
    else:
        for i, data in enumerate(datas):
            axs[i].hist(np.asarray(data).flatten(), bins=n_bins, density=True)
            axs[i].set_title(reduce_label(labels[i]))

    if plot_dir is not None:
        savefig(plot_dir)


def plot_histogram_diff(datas, labels, plot_dir=None):
    for i, (hist_diff, bin_edges_diff) in enumerate(datas):
        plt.figure(figure(), figsize=(10, 6))

        bin_centers_diff = (bin_edges_diff[:-1] + bin_edges_diff[1:]) / 2

        plt.bar(
            bin_centers_diff,
            hist_diff,
            width=bin_edges_diff[1] - bin_edges_diff[0],
            alpha=0.7,
            label="Difference",
        )

        plt.title(f"Histogram of differences for {reduce_label(labels[i])}")
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)

        if plot_dir is not None:
            savefig(plot_dir)


def plot_performance_diagram(
    data,
    labels,
    colors=["red", "blue", "chartreuse"],
    markers=["s", "o", "v"],
    title="Performance diagram",
    plot_dir=None,
):
    plt.close("all")
    plt.figure(figure(), figsize=(9, 8))
    legend_params = dict(loc=4, fontsize=12, framealpha=1, frameon=True)
    csi_cmap = "Blues"
    ticks = np.arange(0, 1.1, 0.1)
    grid_ticks = np.arange(0, 1.01, 0.01)
    xlabel = "Success Ratio (1-FAR)"
    ylabel = "Probability of Detection"
    csi_label = "Critical Success Index"
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)
    csi_contour = plt.contourf(
        sr_g, pod_g, csi, np.arange(0.1, 1.1, 0.1), extend="max", cmap=csi_cmap
    )
    b_contour = plt.contour(
        sr_g, pod_g, bias, [0.5, 1, 1.5, 2, 4], colors="k", linestyles="dashed"
    )
    plt.clabel(
        b_contour, fmt="%1.1f", manual=[(0.2, 0.9), (0.4, 0.9), (0.6, 0.9), (0.7, 0.7)]
    )

    for r, d in enumerate(data):
        pod = d[0]
        far = d[1]
        plt.plot(
            1 - far,
            pod,
            marker=markers[r],
            linewidth=12,
            color=colors[r],
            label=labels[r].replace("_dist", "").replace("-", " ").replace("_", " "),
        )

    cbar = plt.colorbar(csi_contour)
    cbar.set_label(csi_label, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title(title, fontsize=14)
    plt.text(0.48, 0.6, "Frequency Bias", fontdict=dict(fontsize=14, rotation=45))
    plt.legend()

    if plot_dir is not None:
        savefig(plot_dir)


def plot_hist(hist, model_dir=None, show=False, save_path=None, name_files=False):
    return
    if save_path is None:
        save_path = model_dir

    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    plt.title(f"training accuracy for\n{model_dir}")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    if show:
        plt.show()
    else:
        filename = "accuracy.png"
        if name_files:
            filename = "{}_{}".format(model_dir, filename)

        plt.savefig("{}/{}".format(save_path, filename))
        print("Wrote file {}/{}".format(save_path, filename))
    plt.close()
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])
    plt.title(f"training loss for\n{model_dir}")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    if show:
        plt.show()
    else:
        filename = "loss.png"
        if name_files:
            filename = "{}_{}".format(model_dir, filename)

        plt.savefig("{}/{}".format(save_path, filename))
        print("Wrote file {}/{}".format(save_path, filename))


def plot_fss(
    data,
    masks,
    labels,
    obs_frac,
    img_sizes,
    times,
    full_hours_only=False,
    plot_dir=None,
):
    plt.close("all")
    domain_x = 2370  # km
    domain_y = 2670

    CATEGORIES = ["clearsky", "partly-cloudy", "cloudy"]

    factor = 60 if full_hours_only else 15

    # shape: a, b, c, d, e
    # a: model label
    # b: mask
    # c: forecast number
    # d: bin
    # e: leadtime (data)

    for i in range(data.shape[0]):  # labels
        for j in range(data.shape[1]):  # categories (bins)
            cat = CATEGORIES[j]

            plt.figure(figure(), figsize=(8, 8))
            dx = int(np.ceil(domain_x / float(img_sizes[i][0])))
            x = np.arange(data.shape[-1])
            y = np.arange(len(masks))

            xx, yy = np.meshgrid(x, y)

            v = []

            for k, m in enumerate(masks):
                v.append(np.nanmean(data[i][j][k], axis=0))

            v = np.asarray(v)
            v = np.nan_to_num(v)
            v = np.ma.masked_where(v == 0.0, v)

            try:
                levels = np.linspace(0.3, 1.0, 21)
                fss_good = 0.5 + (obs_frac[j] * 0.5)
                print("Good FSS for category {}={:.3f}".format(CATEGORIES[j], fss_good))

                plt.contourf(xx, yy, v, levels=levels)
                plt.colorbar()
                plt.title(
                    "FSS for category '{}' (FSS_good={:.2f})\ntime {}..{} '{}'".format(
                        cat, fss_good, times[0], times[-1], reduce_label(labels[i])
                    )
                )
                plt.xlabel("Leadtime (minutes)")
                plt.ylabel("Mask size (km)")
                plt.xticks(x, list(map(lambda x: "{}".format(x * factor), x)))
                plt.yticks(y, list(map(lambda x: "{}".format(int(x * dx)), masks)))

                CS = plt.contour(xx, yy, v, [fss_good])
                plt.clabel(CS, inline=True, fontsize=10)
            except ValueError as e:
                print("Failed to plot for bin #{}".format(j))
                print(e)
                continue

            if plot_dir is not None:
                savefig(plot_dir)


def plot_psd(scales, psd_values, title, plot_dir=None):
    plt.close("all")
    plt.figure(figure(), figsize=(10, 6))
    for i in range(len(psd_values)):
        # Average PSD over all forecasts and sum over y-scales to get 1D PSD
        psd_mean = psd_values[i].mean(axis=0).sum(axis=-1)
        plt.plot(scales, psd_mean, label=f"Lead Time={i}h")

    psd_values = np.asarray(psd_values)
    ave = np.mean(psd_values, axis=(0, 1)).sum(axis=-1)

    plt.plot(
        scales,
        ave,
        label="Average PSD",
        color="black",
        linestyle="--",
    )
    plt.xlabel("Horizontal Scale [km]")
    plt.ylabel("Power Spectral Density [%²/km]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xscale("log")  # Use logarithmic scale for better visualization
    plt.yscale("log")  # Use logarithmic scale for better visualization of y-axis
    plt.gca().invert_xaxis()  # Invert the x-axis for descending order
    plt.autoscale(True, axis="y")  # Auto-scale the y-axis

    if plot_dir is not None:
        savefig(plot_dir)


def plot_psd_ave(scales, psd_values, title, labels, plot_dir=None):
    plt.close("all")
    plt.figure(figure(), figsize=(10, 6))
    psd_values = np.asarray(psd_values)

    for i, psd_mean in enumerate(psd_values):
        plt.plot(scales, psd_mean, label=f"Mean PSD for {labels[i]}")

    plt.xlabel("Horizontal Scale [km]")
    plt.ylabel("Power Spectral Density [%²/km]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xscale("log")  # Use logarithmic scale for better visualization
    plt.yscale("log")  # Use logarithmic scale for better visualization of y-axis
    plt.gca().invert_xaxis()  # Invert the x-axis for descending order
    plt.autoscale(True, axis="y")  # Auto-scale the y-axis

    if plot_dir is not None:
        savefig(plot_dir)


def plot_chisquare(data, labels, title, plot_dir=None):
    plt.close("all")

    plt.figure(figure())
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = itertools.cycle(["b", "r", "g", "c", "m", "y", "k"])
    markers = itertools.cycle(["o", "s", "D", "^", "v", "<", ">"])

    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis

    for i, forecast in enumerate(data):
        chi_squared, p_values = zip(*forecast)
        color = next(colors)
        marker = next(markers)

        # Plot the chi-squared as a line plot
        ax1.plot(chi_squared, label=f"{labels[i]} Chi-squared", color=color)

        # Plot the p-values as a scatter plot
        ax2.scatter(
            range(len(p_values)),
            p_values,
            label=f"{labels[i]} p-value",
            color=color,
            marker=marker,
        )

    ax2.set_ylim(0, 1.1)  # Set the y-axis limits for the p-values
    # Add labels and legend
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Chi-squared Value")
    ax2.set_ylabel("p-value")

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
    plt.grid(True)

    plt.title("Chi-squared and p-values")

    if plot_dir is not None:
        savefig(plot_dir)
