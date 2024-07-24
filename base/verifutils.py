import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from base.plotutils import *
import tensorflow as tf
from fss import make_FSS_loss
import time

CATEGORIES = ["cloudy", "partly-cloudy", "clear"]


def get_time_for_file(args):
    if args.start_date is None:
        return ""

    if args.start_date == args.stop_date:
        return args.start_date.strftime("%Y%m%dT%H%M%S")

    return "{}-{}".format(
        args.start_date.strftime("%Y%m%d"), args.stop_date.strftime("%Y%m%d")
    )


def produce_scores(args, predictions):
    mae(args, predictions)
    psd(args, predictions)
    # categorical_scores(args, predictions)
    histogram(args, predictions)
    ssim(args, predictions)
    fss(args, predictions)
    print("All scores produced")


def histogram(args, predictions):
    print("Plotting histogram")
    datas = []
    labels = []

    for l in predictions:
        datas.append(predictions[l]["data"])
        labels.append(l)

        if False and args.result_dir is not None:
            np.save(
                "{}/{}_{}_histogram.npy".format(
                    args.result_dir, l, get_time_for_file(args)
                ),
                np.histogram(np.asarray(datas[-1]), bins=50),
            )

    datas = np.asarray(datas, dtype=object)
    plot_histogram(datas, labels, plot_dir=args.plot_dir)


def mae(args, predictions):
    ae, times = ae2d(args, predictions)
    plot_mae_per_leadtime(args, ae)

    ae, times = remove_initial_ae(ae, times)
    plot_mae2d(args, ae, times)
    plot_mae_timeseries(args, ae, times)

    if args.stats_dir is None:
        return

    with open(args.stats_dir + "/stats.txt", "a") as f:
        for label in ae:
            # shape (forecasts, leadtime, x, y, channels)
            for i in range(ae[label].shape[1]):
                f.write(
                    "{} MAE for forecast {}: {:.2f}\n".format(
                        label, i, np.mean(ae[label][:, i])
                    )
                )
            f.write("{} Average MAE value: {:.2f}\n".format(label, np.mean(ae[label])))


def remove_initial_ae(ae, times):
    newae = {}
    for l in ae:
        if l == "persistence":
            continue
        newae[l] = []
        for i, pred_data in enumerate(ae[l]):
            newae[l].append(np.delete(pred_data, 0, axis=0))

        newae[l] = np.asarray(newae[l], dtype=np.float32)

    for i, t in enumerate(times):
        times[i] = t[1:]

    return newae, times


# absolute error 2d
def ae2d(args, predictions):
    print("Producing error fields")
    ret = {}
    gtt = np.asarray(predictions["gt"]["time"])
    gtd = np.asarray(predictions["gt"]["data"])
    times = []
    first = True

    for l in predictions:
        if l == "gt":
            continue

        ret[l] = []

        for pred_times, pred_data in zip(
            predictions[l]["time"], predictions[l]["data"]
        ):
            if args.full_hours_only is False:
                a = np.where(gtt == pred_times[0])[0][0]
                b = np.where(gtt == pred_times[-1])[0][0]
                gt_data = gtd[a : b + 1]
            else:
                idx = np.where(np.isin(gtt, pred_times))
                gt_data = gtd[idx]

            assert gt_data.shape == pred_data.shape

            mae = np.abs(gt_data - pred_data).astype(np.float32)
            ret[l].append(mae)

            if first:
                times.append(pred_times)

        ret[l] = np.asarray(ret[l], dtype=np.float32)

        assert len(ret[l]) == len(times)

        first = False

    # create persistence
    ret["persistence"] = []
    for i, pred_times in enumerate(times):
        if args.full_hours_only is False:
            a = np.where(gtt == pred_times[0])[0][0]
            b = np.where(gtt == pred_times[-1])[0][0]
            gt_data = gtd[a : b + 1]
        else:
            idx = np.where(np.isin(gtt, pred_times))
            gt_data = gtd[idx]

        initial = np.expand_dims(gt_data[0], axis=0)
        initial = np.repeat(initial, len(pred_times), axis=0)

        assert gt_data.shape == initial.shape

        mae = np.abs(gt_data - initial).astype(np.float32)
        ret["persistence"].append(mae)

    ret["persistence"] = np.asarray(ret["persistence"])

    return ret, times


def plot_mae_per_leadtime(args, ae):
    print("Plotting mae per leadtime")

    labels = list(ae.keys())

    maelts = []
    for j, l in enumerate(ae):
        newae = np.moveaxis(ae[l], 0, 1)
        maelt = []

        for i, lt in enumerate(newae):
            maelt.append(np.mean(lt).astype(np.float32))

        maelts.append(np.asarray(maelt, dtype=np.float32))
        labels[j] += " ({:.3f})".format(np.mean(np.asarray(maelt)))

        if args.result_dir is not None:
            np.save(
                "{}/{}_{}_mae.npy".format(args.result_dir, l, get_time_for_file(args)),
                np.asarray(maelt),
            )

    plot_linegraph(
        maelts,
        labels,
        title="MAE over {} predictions".format(ae[l].shape[0]),
        ylabel="mae",
        plot_dir=args.plot_dir,
        start_from_zero=True,
        full_hours_only=args.full_hours_only,
    )


def plot_mae2d(args, ae, times):
    # dimensions of mae2d:
    # (9, 4, 128, 128, 1)
    # (forecasts, leadtimes, x, y, channels)
    # merge 0, 1 so that all mae2d fields are merged to one dimension
    print("Plotting mae on map")

    if times is not None:
        title = "MAE between {}..{}".format(
            times[0][0].strftime("%Y%m%dT%H%M"), times[-1][-1].strftime("%Y%m%dT%H%M")
        )
    else:
        title = "Mean Absolute Error 2D"

    for l in ae:
        label = reduce_label(l)
        titlel = "{}\n{}".format(title, label)
        # calculate average 2d field
        img_size = ae[l][0].shape[1:3]
        mae = np.average(
            ae[l].reshape((-1, img_size[0], img_size[1], 1)), axis=0
        ).astype(np.float32)

        plot_on_map(np.squeeze(mae), title=titlel, plot_dir=args.plot_dir)

        if args is not None and args.result_dir is not None:
            np.save(
                "{}/{}_{}_mae2d.npy".format(
                    args.result_dir, l, get_time_for_file(args)
                ),
                mae,
            )

        # calculate mae 2d field per leadtime
        continue
        mae = np.average(ae[l], axis=0).astype(np.float32)

        plotted_leadtimes = (3, 7, 11)  # should not be hard coded
        if args.full_hours_only:
            plotted_leadtimes = range(mae.shape[0])

        for lt in plotted_leadtimes:  # range(mae.shape[0]):
            plot_on_map(
                np.squeeze(mae[lt]),
                title="MAE leadtime={}m\n{}".format((1 + lt) * 15, label),
                plot_dir=args.plot_dir,
            )

            if args is not None and args.result_dir is not None:
                np.save(
                    "{}/{}_{}_mae2d_{}.npy".format(
                        args.result_dir, l, get_time_for_file(args), lt
                    ),
                    mae,
                )


def plot_mae_timeseries(args, ae, times):
    print("Plotting mae timeseries")

    # if less than leadtime_conditioning forecasts are found for a given time,
    # do not include that to data (because the results are skewed)
    trim_short_times = True

    num_expected_forecasts = args.prediction_len

    if args.full_hours_only:
        num_expected_forecasts = int(args.prediction_len / 4)

    def process_data(ae, times):
        maets = {}

        # produce a dict where each leadtime is a key and value is a list
        # containing all errors for that leadtime (from different forecasts)

        for i, aes in enumerate(ae):
            assert len(times[i]) == len(aes)
            for j, _ae in enumerate(aes):
                t = times[i][j]
                try:
                    maets[t].append(_ae)
                except KeyError as e:
                    maets[t] = [_ae]

        # create x,y,count lists where x = leadtime, y = mean mae for that leadtime,
        # count is number of forecasts for that leadtime

        counts = []
        x = []
        y = []
        for t in maets.keys():
            if trim_short_times and len(maets[t]) < num_expected_forecasts:
                continue
            counts.append(len(maets[t]))
            y.append(np.average(maets[t]).astype(np.float32))
            x.append(t)
        return x, y, counts

    def aggregate_to_max_hour(ae_timeseries, times):
        x, y, counts = process_data(ae_timeseries, times)
        return x, y, counts

    data = []

    for l in ae.keys():
        assert len(times) == len(ae[l])
        mx, my, mcounts = aggregate_to_max_hour(ae[l], times)
        data.append(my)

    xlabels = list(map(lambda x: x.strftime("%H:%M"), mx))

    plot_normal(
        mx,
        data,
        mcounts,
        list(map(lambda x: reduce_label(x), ae.keys())),
        title="MAE between {}..{}".format(
            times[0][0].strftime("%Y%m%dT%H%M"), times[-1][-1].strftime("%Y%m%dT%H%M")
        ),
        xlabels=xlabels,
        plot_dir=args.plot_dir,
    )


def calculate_categorical_score(category, cm, score):
    def calc_score(TN, FP, FN, TP, score):
        if score == "POD":
            return TP / (TP + FN)
        elif score == "FAR":
            return FP / (TP + FP)
        elif score == "CSI":
            SR = 1 - calc_score(TN, FP, FN, TP, "FAR")
            return 1 / (1 / SR + 1 / calc_score(TN, FP, FN, TP, "POD") - 1)

    idx = CATEGORIES.index(category)
    TP = cm[idx, idx]
    FN = (
        np.sum(
            cm[
                idx,
            ]
        )
        - TP
    )
    FP = np.sum(cm[:, idx]) - TP
    TN = (
        np.sum(cm)
        - np.sum(
            cm[
                idx,
            ]
        )
        - np.sum(cm[:, idx])
    )

    return calc_score(TN, FP, FN, TP, score)


def categorize(arr):
    # cloudy = 2 when cloudiness > 85%
    # partly = 1 when 15% >= cloudiness >= 85%
    # clear = 0  when cloudiness < 15%
    return np.digitize(arr, [0.15, 0.85]).astype(np.int8)


def categorical_scores(args, predictions):
    print("Plotting categorical scores")

    gtd = predictions["gt"]["data"]
    gtt = predictions["gt"]["time"]

    gtd = np.asarray(gtd).copy()
    gtd = categorize(gtd)

    keys = list(predictions.keys())
    keys.remove("gt")

    def calc_confusion_matrix(predictions):
        ret = {}
        for l in predictions:
            if l == "gt":
                continue

            ret[l] = {}

            preds = []
            gts = []
            for pred_times, pred_data in zip(
                predictions[l]["time"], predictions[l]["data"]
            ):
                a = gtt.index(pred_times[0])
                b = gtt.index(pred_times[-1])

                gt_data = gtd[a : b + 1]

                pred_data = categorize(pred_data.copy())

                assert gt_data.shape == pred_data.shape

                preds.append(pred_data.flatten())
                gts.append(gt_data.flatten())

            preds = np.asarray(preds)
            gts = np.asarray(gts)

            ret[l] = np.flip(
                confusion_matrix(gts.flatten(), preds.flatten(), normalize="all")
            )  # wikipedia syntax
        return ret

    cm = calc_confusion_matrix(predictions)

    for l in keys:
        catscores = []
        for c in CATEGORIES:
            p = calculate_categorical_score(c, cm[l], "POD")
            f = calculate_categorical_score(c, cm[l], "FAR")
            catscores.append((p, f))

        plot_performance_diagram(
            catscores,
            CATEGORIES,
            title="Performance diagram over {} predictions\n{}".format(
                len(predictions[keys[0]]["data"]), l
            ),
            plot_dir=args.plot_dir,
        )

        if args.result_dir is not None:
            np.save(
                "{}/{}_{}_categoricalscores.npy".format(
                    args.result_dir, l, get_time_for_file(args)
                ),
                np.asarray(catscores),
            )


def ssim(args, predictions):
    print("Plotting SSIM")

    num_expected_forecasts = args.prediction_len

    if args.full_hours_only:
        num_expected_forecasts = int(args.prediction_len / 4)

    ssims = []
    for l in predictions.keys():
        ssims.append([])

        if l == "gt":
            gtd = predictions["gt"]["data"]
            start = 0
            stop = num_expected_forecasts
            while stop < len(gtd):
                gtdata = gtd[start : stop + 1]
                i = 0
                ssims[-1].append([])
                for cur in gtdata[1:]:
                    prev = gtdata[i]
                    i += 1
                    ssims[-1][-1].append(
                        structural_similarity(
                            np.squeeze(prev), np.squeeze(cur), data_range=1.0
                        ).astype(np.float32)
                    )

                start += num_expected_forecasts
                stop += num_expected_forecasts
                ssims[-1][-1] = np.asarray(ssims[-1][-1])
            ssims[-1] = np.average(np.asarray(ssims[-1]), axis=0).astype(np.float32)
            continue

        for pred_data in predictions[l]["data"]:
            i = 0
            ssims[-1].append([])
            for cur in pred_data[1:]:
                prev = pred_data[i]
                i += 1
                ssims[-1][-1].append(
                    structural_similarity(
                        np.squeeze(prev), np.squeeze(cur), data_range=1.0
                    ).astype(np.float32)
                )

            ssims[-1][-1] = np.asarray(ssims[-1][-1])

        ssims[-1] = np.average(np.asarray(ssims[-1]), axis=0).astype(np.float32)

        if False and args.result_dir is not None:
            np.save(
                "{}/{}_{}_ssim.npy".format(args.result_dir, l, get_time_for_file(args)),
                ssims[-1],
            )

    plot_linegraph(
        ssims,
        list(predictions.keys()),
        title="Mean SSIM over {} predictions".format(
            len(predictions[list(predictions.keys())[0]]["data"])
        ),
        ylabel="ssim",
        plot_dir=args.plot_dir,
        full_hours_only=args.full_hours_only,
    )


def fss(args, predictions):
    def calc_fss(lf, pred_times, pred_data, gtt, gtd):
        arr = []

        for pred_times, pred_data in zip(
            predictions[l]["time"], predictions[l]["data"]
        ):
            if args.full_hours_only is False:
                a = np.where(gtt == pred_times[0])[0][0]
                b = np.where(gtt == pred_times[-1])[0][0]
                gt_data = gtd[a : b + 1]
            else:
                idx = np.where(np.isin(gtt, pred_times))
                gt_data = gtd[idx]

            assert gt_data.shape == pred_data.shape

            datas = []
            for y_true, y_pred in zip(gt_data, pred_data):
                # fss is implemented as a loss functions, so it needs
                # batch dimension
                y_true = np.expand_dims(y_true, 0)
                y_pred = np.expand_dims(y_pred, 0)

                loss = 1 - lf(y_true, y_pred).numpy()
                datas.append(loss)

            assert (args.full_hours_only and len(datas) == len(pred_times)) or (
                args.full_hours_only is False and len(datas) == args.prediction_len + 1
            )

            arr.append(datas)
        return arr

    obs_cat = categorize(predictions["gt"]["data"])
    observed_cat0 = np.count_nonzero(obs_cat == 0)
    observed_cat1 = np.count_nonzero(obs_cat == 1)
    observed_cat2 = np.count_nonzero(obs_cat == 2)

    obs_frac_cat0 = observed_cat0 / (observed_cat0 + observed_cat1 + observed_cat2)
    obs_frac_cat1 = observed_cat1 / (observed_cat0 + observed_cat1 + observed_cat2)
    obs_frac_cat2 = observed_cat2 / (observed_cat0 + observed_cat1 + observed_cat2)

    obs_frac = [obs_frac_cat0, obs_frac_cat1, obs_frac_cat2]

    bins = tf.constant([[0, 0.15], [0.15, 0.85], [0.85, 1.01]], dtype=tf.float32)

    masks = [3, 5, 9, 13, 17, 27, 45, 60, 80]
    labels = list(predictions.keys())
    labels.remove("gt")

    fsss = []
    img_sizes = []

    print("Producing FSS for mask sizes: {}".format(masks))

    for l in predictions:
        if l == "gt":
            continue

        img_sizes.append(predictions[l]["data"][0].shape[1:3])

        label_arr = []
        for m in masks:
            start = time.time()
            lf = make_FSS_loss(m, bins=bins, hard_discretization=True)
            mask_arr = calc_fss(
                lf,
                predictions[l]["time"],
                predictions[l]["data"],
                np.asarray(predictions["gt"]["time"]),
                np.asarray(predictions["gt"]["data"]),
            )
            label_arr.append(mask_arr)
            stop = time.time()
            print("FSS for: {} mask size: {} in {:.1f}s".format(l, m, stop - start))

        fsss.append(label_arr)

    fsss = np.asarray(fsss)
    fsss = np.moveaxis(fsss, -1, 1)

    # shape: a, b, c, d, e
    # a: model label
    # b: bin
    # c: mask
    # d: forecast number
    # e: leadtime

    plot_fss(
        fsss,
        masks,
        labels,
        obs_frac,
        img_sizes,
        predictions["gt"]["time"],
        plot_dir=args.plot_dir,
        full_hours_only=args.full_hours_only,
    )


def calculate_psd(data, args):

    """
    Calculate Power Spectral Density over the spatial dimensions.
    :param data: 2D numpy array
    :param dx: sampling interval in spatial dimensions
    :return: spatial frequency and PSD values
    """

    def apply_window(data):
        """
        Apply a Hanning window to the data.
        :param data: 2D numpy array
        :return: windowed data
        """
        nx, ny = data.shape[-2:]
        window_x = np.hanning(nx)
        window_y = np.hanning(ny)
        window = window_x[:, np.newaxis] * window_y[np.newaxis, :]
        return data * window

    data = apply_window(data)

    # Number of spatial points
    nx, ny = data.shape[-2:]
    dx = 2500 * (949 - 1.0) / nx / 1000

    # Perform Fourier transform over spatial dimensions
    f_transform = np.fft.fft2(data, axes=(-2, -1))
    f_transform = np.fft.fftshift(f_transform)
    # Calculate power spectral density
    psd = np.abs(f_transform) ** 2
    # Calculate spatial frequencies
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dx)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    # Convert spatial frequencies to physical scales in kilometers
    scale_x = (
        1 / kx[nx // 2 + 1 :]
    )  # Ignore the negative frequencies and zero frequency
    scale_y = (
        1 / ky[ny // 2 + 1 :]
    )  # Ignore the negative frequencies and zero frequency

    # Reverse the scales for descending order
    scale_x = scale_x[::-1]
    scale_y = scale_y[::-1]

    psd = psd[:, nx // 2 + 1 :, ny // 2 + 1 :][:, ::-1, ::-1]

    return scale_x, scale_y, psd


def psd(args, predictions):
    psd_values = []

    for l in predictions:
        if l == "gt":
            continue

        data = np.asarray(predictions[l]["data"])
        data = np.squeeze(data, axis=-1)

        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=1)

        for i in range(4, 22, 4):  # pick only full hour data
            scale_x, scale_y, psd = calculate_psd(data[:, i], args)
            psd_values.append(psd)

        plot_psd(
            scale_x,
            psd_values,
            list(range(1, 6)),
            plot_dir=args.plot_dir,
        )

        if args.stats_dir is None:
            return

        with open(args.stats_dir + "/stats.txt", "a") as f:
            for i, psd in enumerate(psd_values):
                f.write(
                    "{} PSD value for forecast {}: {:.2f}\n".format(l, i, np.mean(psd))
                )
            f.write("{} Average PSD value: {:.2f}\n".format(l, np.mean(psd_values)))
