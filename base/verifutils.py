import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from base.plotutils import *
import tensorflow as tf
from fss import make_FSS_loss
import time
from scipy.stats import chisquare
import pywt

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

    for score in args.scores:
        if score == "mae":
            print("Calculating MAE")
            mae(args, predictions)
        elif score == "psd":
            print("Calculating PSD")
            psd(args, predictions)
        elif score == "categorical":
            print("Calculating categorical scores")
            categorical_scores(args, predictions)
        elif score == "chi_squared":
            print("Calculating chi-squared")
            chi_squared(args, predictions)
        elif score == "change":
            print("Calculating change")
            change(args, predictions)
        elif score == "histogram":
            print("Calculating histogram")
            histogram(args, predictions)
        elif score == "ssim":
            print("Calculating SSIM")
            ssim(args, predictions)
        elif score == "fss":
            print("Calculating FSS")
            fss(args, predictions)
        elif score == "wavelet":
            print("Calculating wavelet scores")
            wavelet_scores(args, predictions)
        elif score == "maess":
            print("Calculating MAESS")
            maess(args, predictions)

    print("All scores produced")


def maess(args, predictions):

    gtd = np.asarray(predictions["gt"]["data"])
    ae, times = ae2d(args, predictions)

    n_samples, n_leadtimes, n_x, n_y, _ = ae["persistence"].shape

    def monte_carlo_reference_forecast(observations, num_simulations=50):
        n_samples, n_features = observations.shape
        simulated_maes = []

        for _ in range(num_simulations):
            simulated_forecast_flat = np.random.choice(
                observations.flatten(), size=n_features, replace=True
            )
            simulated_forecast = simulated_forecast_flat.reshape(n_x, n_y, 1)
            simulated_mae = np.mean(
                np.abs(
                    simulated_forecast - observations.reshape(n_samples, n_x, n_y, 1)
                )
            )

            simulated_maes.append(simulated_mae)

        return np.mean(simulated_maes)

    reference_mae = monte_carlo_reference_forecast(gtd.reshape(gtd.shape[0], -1))

    maesslts = []
    selts = []

    for i, l in enumerate(ae.keys()):
        if l == "persistence":
            continue

        maesslt = []
        selt = []

        for leadtime in range(ae[l].shape[1]):
            mae = np.mean(ae[l][:, leadtime])
            se = np.std(ae[l][:, leadtime]) / np.sqrt(n_samples)
            maess = 1 - (mae / reference_mae)
            maesslt.append(maess)
            selt.append(se)

        maesslts.append(maesslt)
        selts.append(selt)

    labels = [reduce_label(l) for l in ae.keys() if l != "persistence"]

    if args.stats_dir is not None:
        with open(args.stats_dir + "/stats.txt", "a") as f:
            f.write("Reference MAE: {:.3f}\n".format(reference_mae))
            for i, (maes, ses) in enumerate(zip(maesslts, selts)):
                l = reduce_label(labels[i])
                for lt, (mae, se) in enumerate(zip(maes, ses)):
                    f.write(
                        "{} Leadtime: {} MAESS: {:.3f} SE: {:.4f}\n".format(
                            l, lt, mae, se
                        )
                    )

                f.write(f"{l} Mean MAESS: {np.mean(maes):.3f}\n")

    plot_linegraph(
        maesslts,
        labels,
        title="MAE Skill Score",
        ylabel="MAESS",
        plot_dir=args.plot_dir,
        start_from_zero=True,
        full_hours_only=(args.full_hours_only or args.hourly_data),
        errors=selts,
    )


def wavelet_scores(args, predictions):
    def summarize_wavelet(HH, measure="L1"):
        if measure == "L1":
            return np.sum(np.abs(HH))
        elif measure == "L2":
            return np.sum(HH**2)
        elif measure == "entropy":
            HH_flat = HH.flatten()
            p = np.abs(HH_flat) / np.sum(np.abs(HH_flat))
            p = p[p > 0]  # Avoid log(0)
            return -np.sum(p * np.log2(p))
        else:
            raise ValueError("Unknown measure type")

    gtd = np.asarray(predictions["gt"]["data"])
    gtt = np.asarray(predictions["gt"]["time"])

    all_sad = {}
    all_energy = {"gt": []}
    gt_done = False
    for i, l in enumerate(predictions):
        if l == "gt":
            continue

        all_sad[l] = []
        all_energy[l] = []

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

            # leadtime, x, y, channels
            assert gt_data.shape == pred_data.shape

            sad = []
            energy = []
            energy_gt = []
            for lt in range(gt_data.shape[0]):
                image_gt = gt_data[lt].squeeze(axis=-1)
                image_fc = pred_data[lt].squeeze(axis=-1)

                coeffs2_gt = pywt.dwt2(image_gt, "haar")
                coeffs2_fc = pywt.dwt2(image_fc, "haar")

                _, (_, _, HH_gt) = coeffs2_gt
                _, (_, _, HH_fc) = coeffs2_fc

                energy.append(summarize_wavelet(HH_fc, measure="L2"))

                if gt_done is False:
                    energy_gt.append(summarize_wavelet(HH_gt, measure="L2"))

                # Calculate Sum of Absolute Differences in Wavelet HH Coefficients
                sad_wavelet_HH = np.sum(np.abs(HH_gt - HH_fc))
                sad.append(sad_wavelet_HH)

            all_sad[l].append(sad)
            all_energy[l].append(energy)
            if gt_done is False:
                all_energy["gt"].append(energy_gt)

        gt_done = True

    sads = []
    for l in all_sad:
        all_sad[l] = np.asarray(all_sad[l], dtype=np.float32)
        sads.append(np.mean(all_sad[l], axis=0))
    energys = []
    for l in all_energy:
        all_energy[l] = np.asarray(all_energy[l], dtype=np.float32)
        energys.append(np.mean(all_energy[l], axis=0))

    plot_linegraph(
        sads,
        [reduce_label(l) for l in all_sad.keys()],
        title="Sum of absolute differences in wavelet HH coefficients",
        ylabel="SAD Wavelet HH difference",
        plot_dir=args.plot_dir,
        start_from_zero=True,
        full_hours_only=(args.full_hours_only or args.hourly_data),
    )
    plot_linegraph(
        energys,
        [reduce_label(l) for l in all_energy.keys()],
        title="Wavelet entropy",
        ylabel="Entropy",
        plot_dir=args.plot_dir,
        start_from_zero=True,
        full_hours_only=(args.full_hours_only or args.hourly_data),
    )

    if args.stats_dir is not None:
        labels = [reduce_label(l) for l in all_sad.keys()]
        with open(args.stats_dir + "/stats.txt", "a") as f:
            for i, s in enumerate(sads):
                l = reduce_label(labels[i])
                for lt, v in enumerate(s):
                    f.write(
                        "{} Leadtime: {} Sum of absolute differences in wavelet HH coefficients: {:.1f}\n".format(
                            l, lt, v
                        )
                    )

                f.write(f"{l} Mean SAD sum: {np.sum(s):.1f}\n")

        with open(args.stats_dir + "/stats.txt", "a") as f:
            labels = [reduce_label(l) for l in all_energy.keys()]
            for i, s in enumerate(energys):
                l = reduce_label(labels[i])
                for lt, v in enumerate(s):
                    f.write(
                        "{} Leadtime: {} Wavelet entropy: {:.1f}\n".format(l, lt, v)
                    )

                f.write(f"{l} Mean Entropy sum: {np.sum(s):.1f}\n")


def change(args, predictions):
    gtd = np.asarray(predictions["gt"]["data"])
    gtt = np.asarray(predictions["gt"]["time"])

    all_diff = {}
    for i, l in enumerate(predictions):
        if l == "gt":
            continue

        all_diff[l] = []

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

            # leadtime, x, y, channels
            assert gt_data.shape == pred_data.shape

            # calculate change in data
            diff = np.mean(gt_data - pred_data, axis=(1, 2, 3)).astype(np.float32)

            all_diff[l].append(diff)

    pl = []
    for l in all_diff:
        all_diff[l] = np.asarray(all_diff[l], dtype=np.float32)
        pl.append(np.mean(all_diff[l], axis=0))

    labels = [reduce_label(l) for l in all_diff.keys()]

    plot_linegraph(
        pl,
        labels,
        title="Mean difference to ground truth",
        ylabel="Difference",
        plot_dir=args.plot_dir,
        start_from_zero=True,
        full_hours_only=(args.full_hours_only or args.hourly_data),
    )


def chi_squared(args, predictions):
    gtd = np.asarray(predictions["gt"]["data"])
    gtt = np.asarray(predictions["gt"]["time"])

    all_data = []
    for l in predictions:
        if l == "gt":
            continue

        ret = []

        gt_lt = {}
        pred_lt = {}

        n = predictions[l]["data"][0].shape[0]
        for i in range(n):
            gt_lt[i] = []
            pred_lt[i] = []

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

            assert (
                gt_data.shape == pred_data.shape
            ), "shapes do not match for gt and {}: {} vs {}".format(
                l, gt_data.shape, pred_data.shape
            )

            for i in range(pred_data.shape[0]):
                gt_lt[i].append(gt_data[i].flatten())
                pred_lt[i].append(pred_data[i].flatten())

        bins = np.linspace(0, 1, 21)  # Define bins for range [0, 1]

        for i in range(n):
            gt_data = np.asarray(gt_lt[i]).flatten()
            pred_data = np.asarray(pred_lt[i]).flatten()

            hist_gtd, _ = np.histogram(gt_data, bins=bins)
            hist_pred, _ = np.histogram(pred_data, bins=bins)

            # Add small constant to histogram counts to avoid zero expected frequencies
            hist_pred = hist_pred + 1e-10

            # Normalize histograms to have the same sum
            hist_gtd = hist_gtd / np.sum(hist_gtd)
            hist_pred = hist_pred / np.sum(hist_pred)

            chi2_statistic, p_val_chisquare = chisquare(hist_gtd, f_exp=hist_pred)

            ret.append((chi2_statistic, p_val_chisquare))

        if args.stats_dir is not None:
            with open(args.stats_dir + "/stats.txt", "a") as f:
                chis = []
                for lt, v in enumerate(ret):
                    chis.append(v[0])
                    f.write(
                        "{} Leadtime: {} Chi-squared: {:.3f} p-value: {:.3f}\n".format(
                            reduce_label(l), lt, v[0], v[1]
                        )
                    )

                f.write(
                    f"{reduce_label(l)} Summed chi-square values: {np.sum(np.asarray(chis)):.3f}\n"
                )

        all_data.append(ret)

    labels = [reduce_label(l) for l in predictions.keys()]
    plot_chisquare(all_data, labels, f"Chi-squared values", plot_dir=args.plot_dir)


def histogram(args, predictions):
    datas = []
    labels = []

    for l in predictions:
        datas.append(predictions[l]["data"])
        labels.append(l)

    datas = np.asarray(datas, dtype=object)
    plot_histogram(datas, labels, plot_dir=args.plot_dir)

    gtd = np.asarray(predictions["gt"]["data"])
    gtt = np.asarray(predictions["gt"]["time"])

    ret = []
    for l in predictions:
        if l == "gt":
            continue

        diff = []
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

            assert (
                gt_data.shape == pred_data.shape
            ), "shapes do not match for gt and {}: {} vs {}".format(
                l, gt_data.shape, pred_data.shape
            )

            diff.append(gt_data - pred_data)

        hist_diff, bin_edges_diff = np.histogram(diff, bins=20, density=False)

        ret.append((hist_diff, bin_edges_diff))

    plot_histogram_diff(ret, labels, plot_dir=args.plot_dir)


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
                    "{} MAE for forecast {}: {:.3f}\n".format(
                        reduce_label(label), i, np.mean(ae[label][:, i])
                    )
                )
            f.write(
                "{} Average MAE value: {:.3f}\n".format(
                    reduce_label(label), np.mean(ae[label])
                )
            )


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
    labels = list(ae.keys())

    maelts = []
    selts = []
    for j, l in enumerate(ae):
        newae = np.moveaxis(ae[l], 0, 1)
        maelt = []
        selt = []
        for i, lt in enumerate(newae):
            maelt.append(np.mean(lt).astype(np.float32))
            selt.append(np.std(lt) / np.sqrt(len(lt)))

        maelts.append(np.asarray(maelt, dtype=np.float32))
        selts.append(np.asarray(selt, dtype=np.float32))
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
        full_hours_only=(args.full_hours_only or args.hourly_data),
        errors=selts,
    )


def plot_mae2d(args, ae, times):
    # dimensions of mae2d:
    # (9, 4, 128, 128, 1)
    # (forecasts, leadtimes, x, y, channels)
    # merge 0, 1 so that all mae2d fields are merged to one dimension

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
    # if less than leadtime_conditioning forecasts are found for a given time,
    # do not include that to data (because the results are skewed)
    # disable if additional forecasts are included, because then we have less
    # data to work with

    trim_short_times = args.include_additional is None

    first_key = next(iter(ae))
    num_expected_forecasts = ae[first_key].shape[1]

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


def ssim(args, predictions):
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

    plot_linegraph(
        ssims,
        list(predictions.keys()),
        title="Mean SSIM over {} predictions".format(
            len(predictions[list(predictions.keys())[0]]["data"])
        ),
        ylabel="ssim",
        plot_dir=args.plot_dir,
        full_hours_only=(args.full_hours_only or args.hourly),
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

            assert (
                (args.full_hours_only or args.hourly_data)
                and len(datas) == len(pred_times)
            ) or (
                (args.full_hours_only or args.hourly_data) is False
                and len(datas) == args.prediction_len + 1
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

    bins = tf.constant([[0, 0.125], [0.125, 0.875], [0.875, 1.01]], dtype=tf.float32)

    masks = [3, 5, 9, 13, 19, 29, 45, 63]
    labels = list(predictions.keys())
    labels.remove("gt")

    fsss = []
    img_sizes = []

    #    print("Producing FSS for mask sizes: {}".format(masks))

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

            if args.stats_dir is None:
                continue

            with open(args.stats_dir + "/stats.txt", "a") as f:
                mask_arr = np.asarray(mask_arr)  # (predictions, leadtime, category)
                for cat in range(mask_arr.shape[2]):
                    for lt in range(mask_arr.shape[1]):
                        f.write(
                            "{} FSS cat {} mask size {} leadtime {}: {:.3f}\n".format(
                                reduce_label(l),
                                cat,
                                m,
                                lt,
                                np.nanmean(mask_arr[:, lt, cat]),
                            )
                        )
                    f.write(
                        "{} FSS cat {} mask size {} mean value: {:.2f}\n".format(
                            reduce_label(l), cat, m, np.mean(mask_arr[:, :, cat])
                        )
                    )

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

    all_psds = []

    for l in predictions:
        psd_values = []
        data = np.asarray(predictions[l]["data"])
        data = np.squeeze(data, axis=-1)

        if l == "gt":
            scale_x, scale_y, psd_values = calculate_psd(data, args)
            all_psds.append(np.mean(psd_values, axis=0).sum(axis=-1))
            assert np.isnan(psd_values).any() == False

            continue

        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=1)

        leadtimes = range(args.prediction_len + 1)

        if args.full_hours_only:
            leadtimes = range(4, 22, 4)
        if args.hourly_data:
            leadtimes = range(data.shape[1])

        for i in leadtimes:
            scale_x, scale_y, psd = calculate_psd(data[:, i], args)
            psd_values.append(psd)

        plot_psd(
            scale_x,
            psd_values,
            f"Power Spectral Density for {reduce_label(l)}",
            plot_dir=args.plot_dir,
        )

        all_psds.append(np.mean(psd_values, axis=(0, 1)).sum(axis=-1))

        if args.stats_dir is None:
            continue

        with open(args.stats_dir + "/stats.txt", "a") as f:
            for i, psd in enumerate(psd_values):
                f.write(
                    "{} PSD value for forecast {}: {:.2f}\n".format(
                        reduce_label(l), i, np.mean(psd)
                    )
                )
            f.write(
                "{} Average PSD value: {:.2f}\n".format(
                    reduce_label(l), np.mean(psd_values)
                )
            )
    labels = list(predictions.keys())
    labels = [reduce_label(l) for l in labels]

    plot_psd_ave(
        scale_x,
        all_psds,
        "Average Power Spectral Density",
        labels,
        plot_dir=args.plot_dir,
    )
