from tensorflow.keras.models import load_model
from model import *
import glob
import numpy as np
import matplotlib as mpl
import tensorflow_datasets as tfds
import time

# save plots as fiels when running inside a screen instance
mpl.use("Agg")
import matplotlib.pyplot as plt
import argparse

from dateutil import parser as dateparser
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from base.fileutils import *
from base.preprocess import *
from base.plotutils import *
from base.generators import *
from base.verifutils import *
from base.opts import CloudCastOptions
from base.dataseries import LazyDataSeries


def parse_time(timestr):
    masks = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
    for m in masks:
        try:
            return datetime.datetime.strptime(timestr, m)
        except ValueError as e:
            pass

    return None


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action="store", type=str, required=False)
    parser.add_argument("--stop_date", action="store", type=str, required=False)
    parser.add_argument("--dataseries_file", action="store", type=str, required=False)
    parser.add_argument("--label", action="store", nargs="+", type=str, required=True)
    parser.add_argument("--prediction_len", action="store", type=int, default=12)
    parser.add_argument("--include_additional", action="store", nargs="+", default=None)
    parser.add_argument(
        "--plot_dir",
        action="store",
        type=str,
        default=None,
        help="save plots to directory of choice",
    )
    parser.add_argument(
        "--result_dir",
        action="store",
        type=str,
        default=None,
        help="save results to directory of choice",
    )
    parser.add_argument(
        "--stats_dir",
        action="store",
        type=str,
        default=None,
        help="save (some) statistics as txt file to directory of choice",
    )
    parser.add_argument(
        "--prediction_file",
        action="store",
        nargs="+",
        type=str,
        default=None,
        help="read predictions from file, must be one file per label",
    )
    parser.add_argument(
        "--scores",
        action="store",
        nargs="+",
        type=str,
        default=[
            "mae",
            "psd",
            "chi_squared",
            "change",
            "histogram",
            "fss",
            "wavelet",
            "maess",
        ],
        help="list of scores to compute, default is mae, psd, chi_squared, change, histogram, fss, wavelet, maess, other options are: categorical, ssim",
    )
    parser.add_argument(
        "--full_hours_only",
        action="store_true",
        default=False,
        help="Only predict for full hours (leadtime=1h,2h,...)",
    )
    parser.add_argument(
        "--hourly_prediction_only",
        action="store_true",
        default=False,
        help="Make a prediction every hour, in 15 min intervals",
    )
    parser.add_argument(
        "--hourly_data",
        action="store_true",
        default=False,
        help="Data is hourly. Used in conjunction with --prediction_file",
    )

    args = parser.parse_args()

    if (
        (args.start_date is None and args.stop_date is None)
        and args.dataseries_file is None
        and args.prediction_file is None
    ):
        print(
            "One of: (start_date, stop_date), (dataseries_file), (prediction_file) must be given"
        )
        sys.exit(1)

    if args.start_date is not None and args.stop_date is not None:
        args.start_date = parse_time(args.start_date)
        args.stop_date = parse_time(args.stop_date)

    assert args.prediction_file is None or len(args.prediction_file) == len(
        args.label
    ), "Need one prediction file per label"

    assert args.full_hours_only is False or (
        args.full_hours_only and args.prediction_file is not None
    ), "--prediction_file is required with --full_hours_only"

    return args


def normalize_label(label):
    normalized_labels = []

    for lbl in label:
        if lbl.find("*") != -1:
            lbls = [os.path.basename(x) for x in glob.glob(f"models/{lbl}")]
            normalized_labels.extend(lbls)

        else:
            normalized_labels.append(lbl)

    return normalized_labels


def predict_many(args, opts_list):
    all_pred = {}

    for opts in opts_list:
        predictions = predict(args, opts)

        for i, k in enumerate(predictions.keys()):
            if not k in all_pred.keys():
                all_pred[k] = predictions[k]

    return all_pred


def save_predictions(result_dir, label, predictions):
    with open(f"{result_dir}/{label}.npz", "wb") as f:
        np.savez(f, predictions)


def predict(args, opts):
    start = time.time()

    lds = LazyDataSeries(
        opts=opts,
        operating_mode="VERIFY",
        shuffle_data=False,
        reuse_y_as_x=True,
        enable_cache=True,
        enable_debug=True,
        **vars(args),
        leadtime_conditioning=args.prediction_len,
        hourly_prediction=args.hourly_prediction_only,
    )

    d = lds.get_dataset()

    img_size = get_img_size(opts.preprocess)

    model_file = "models/{}".format(opts.get_label())
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    predictions = {
        opts.get_label(): {"time": [], "data": []},
        "gt": {"time": [], "data": []},
    }

    forecast = []
    times = []

    num_expected_forecasts = 1 + args.prediction_len

    if args.full_hours_only:
        num_expected_forecasts = 1 + int(args.prediction_len / 4)

    for t in tfds.as_numpy(d):
        x = t[0]
        y = t[1]
        xy_times = np.squeeze(t[2])
        xy_times = list(map(lambda x: x.decode("utf8"), xy_times))

        prediction_time = datetime.datetime.strptime(xy_times[-1], "%Y%m%dT%H%M%S")

        if args.hourly_prediction_only and xy_times[-2][-4:] != "0000":
            continue

        if len(forecast) > 0 and len(forecast) % num_expected_forecasts == 0:
            predictions[opts.get_label()]["time"].append(times.copy())
            predictions[opts.get_label()]["data"].append(np.array(forecast))

            times.clear()
            forecast.clear()

        if len(forecast) == 0:
            # ground truth as leadtime zero
            x0 = np.expand_dims(np.squeeze(x[..., lds.n_channels - 1], 0), axis=-1)
            t0 = datetime.datetime.strptime(xy_times[-2], "%Y%m%dT%H%M%S")

            forecast.append(x0)
            times.append(t0)

            if t0 not in predictions["gt"]["time"]:
                predictions["gt"]["time"].append(t0)
                predictions["gt"]["data"].append(x0)

        if args.full_hours_only:
            lt = prediction_time - times[0]
            if int(lt.total_seconds() % 3600) != 0:
                continue

        print(
            "Using {} to predict {} [{}/{}]".format(
                xy_times[:-1], xy_times[-1], len(forecast), num_expected_forecasts - 1
            )
        )

        assert len(forecast) <= num_expected_forecasts

        prediction = m.predict(x, verbose=0)
        prediction = np.squeeze(prediction, axis=0)
        forecast.append(prediction)
        times.append(prediction_time)

        if prediction_time not in predictions["gt"]["time"]:
            predictions["gt"]["time"].append(prediction_time)
            predictions["gt"]["data"].append(np.squeeze(y, axis=0))

    if len(forecast) == num_expected_forecasts:
        predictions[opts.get_label()]["time"].append(times.copy())
        predictions[opts.get_label()]["data"].append(np.array(forecast))

    n_forecast = len(predictions[opts.get_label()]["data"])

    print("Read {} forecasts in {:.1f} sec".format(n_forecast, time.time() - start))

    if n_forecast == 0:
        sys.exit(1)

    if args.result_dir is not None:
        save_predictions(args.result_dir, opts.get_label(), predictions)

    return predictions


def copy_range(gt, start, stop):
    a = gt["time"].index(start)
    b = gt["time"].index(stop)
    return np.asarray(gt["data"][a : b + 1])  # need inclusive end


def sort_errors(errors, best_first=True):
    assert best_first

    labels = list(errors.keys())

    maes = {}

    for l in labels:
        maes[l] = np.mean(errors[l])

    # sort best first
    maes = dict(sorted(maes.items(), key=lambda item: item[1]))

    return list(maes.keys())


def filter_top_n(predictions, errors, n, keep=[]):
    if n == -1 or len(predictions) < n:
        return predictions, errors

    labels = sort_errors(errors)

    print("Filtering predictions from {} to {}".format(len(predictions), n))

    labels = labels[:n]

    for k in keep:
        labels.reverse()
        if k not in labels:
            for i, j in enumerate(labels):
                if j not in keep:
                    labels[i] = k
                    break
        labels.reverse()

    f_predictions = {}
    f_errors = {}

    for l in labels:
        predkey = l
        if l == "persistence":
            predkey = "gt"
        f_predictions[predkey] = predictions[predkey]
        f_errors[l] = errors[l]

    return f_predictions, f_errors


def plot_timeseries(args, predictions):
    while True:
        first = list(predictions.keys())[np.random.randint(len(predictions))]
        if first != "gt":
            break

    if len(predictions) >= 8:
        print("Too many predictions ({}) for timeseries plot".format(len(predictions)))
        return

    labels = list(predictions.keys())
    labels.sort()
    idx = np.random.randint(len(predictions[first]["data"]))

    data = []
    times = predictions[first]["time"][idx]

    assert len(predictions["gt"]["data"]) == len(predictions["gt"]["time"])

    for l in labels:
        if l == "gt":
            if args.full_hours_only:
                temp_data = []
                for t in times:
                    idx_ = predictions["gt"]["time"].index(t)
                    temp_data.append(predictions["gt"]["data"][idx_])

                data.append(np.asarray(temp_data))
            else:
                data.append(copy_range(predictions["gt"], times[0], times[-1]))

            continue
        data.append(predictions[l]["data"][idx])

    labels = list(map(lambda x: reduce_label(x), labels))
    plot_stamps(
        data,
        labels,
        title="Prediction for t0={}".format(times[0]),
        initial_data=None,
        start_from_zero=True,
        full_hours_only=args.full_hours_only,
        plot_dir=args.plot_dir,
    )


def intersect_and_filter(args, dicts):
    labels = [x for x in dicts.keys() if x != "gt"]

    assert len(labels) >= 2, "At least two models and ground truth are required"

    # Extract time and data lists for all models
    times = {label: dicts[label]["time"] for label in labels}
    data = {label: dicts[label]["data"] for label in labels}

    time_c = dicts["gt"]["time"]
    data_c = dicts["gt"]["data"]

    # Find common times across all models

    # required length is N; 21 = 5 hours in 15 min steps
    N = args.prediction_len + 1
    if args.hourly_data:
        N = int(np.ceil(N / 4))

    common_times = set(tuple(t) for t in times[labels[0]] if len(t) >= N)

    for label in labels[1:]:
        common_times &= set(tuple(t) for t in times[label] if len(t) >= N)

    common_times = [list(t) for t in common_times]  # Convert back to list
    common_times = sorted(common_times)

    print(
        "Found {} common forecasts across {} models".format(
            len(common_times), len(labels)
        )
    )

    assert len(common_times) > 0, "No common forecasts found"

    # Filter data based on common times
    def filter_model_data(model_data, common_times):
        filtered_data = {"data": [], "time": []}
        common_times_set = set(tuple(t) for t in common_times)
        for t, d in zip(model_data["time"], model_data["data"]):
            if tuple(t) in common_times_set:
                filtered_data["time"].append(t)
                filtered_data["data"].append(d)
        return filtered_data

    filtered_dict = {
        label: filter_model_data(dicts[label], common_times) for label in labels
    }

    for label in labels:
        assert len(filtered_dict[label]["data"]) == len(common_times)

    flat_time = sorted(list(set(item for sublist in common_times for item in sublist)))

    filtered_data = {"data": [], "time": []}
    for time, data in zip(time_c, data_c):
        if time in flat_time:
            filtered_data["time"].append(time)
            filtered_data["data"].append(data)
        else:
            print(f"Skipping ground truth for {time}: not in forecasted times")

    filtered_dict["gt"] = filtered_data
    gt_set = set(filtered_data["time"])

    for l in labels:
        filtered_times = []
        filtered_data = []
        model = filtered_dict[l]
        for times, data in zip(model["time"], model["data"]):
            # Check if all times in the sublist are in gt_set
            if all(time in gt_set for time in times):
                # If all times are valid, add them to the filtered lists
                filtered_times.append(times)
                filtered_data.append(data)
            else:
                print(
                    "Skipping {} forecast with atime {}: not in ground truth".format(
                        l, times[0]
                    )
                )
        # Update the model dictionary with the filtered lists
        filtered_dict[l]["time"] = filtered_times
        filtered_dict[l]["data"] = filtered_data

    for label in labels:
        assert len(filtered_dict[label]["data"]) > 0

    common_times = list(
        set([item for sublist in filtered_dict[labels[0]]["time"] for item in sublist])
    )

    # Reverse check: are all common times in the ground truth?
    for t in common_times:
        assert (
            t in filtered_dict["gt"]["time"]
        ), f"Forecast time {t} not found in ground truth"

    return filtered_dict


def amend_exim_with_analysis_time(gt, exim):

    new_times = []
    new_data = []

    for i, (times, datas) in enumerate(zip(exim["time"], exim["data"])):
        times = [datetime.datetime.strptime(t, "%Y%m%dT%H%M%S") for t in times]
        analysis_time = times[0] - timedelta(minutes=15)
        analysis_data = np.expand_dims(
            gt["data"][gt["time"].index(analysis_time)], axis=0
        )

        new_times.append([analysis_time] + times)
        new_data.append(np.concatenate((analysis_data, datas), axis=0))

        assert len(new_times[-1]) == 5
        assert new_data[-1].shape[0] == 5
    return {"time": new_times, "data": new_data}


if __name__ == "__main__":
    args = parse_command_line()

    labels = normalize_label(args.label)
    opts_list = []
    for l in labels:
        opts_list.append(CloudCastOptions(label=l))
        assert opts_list[-1].onehot_encoding is False

    if args.prediction_file is not None and len(args.prediction_file) > 0:
        predictions = {}
        for i, f in enumerate(args.prediction_file):
            assert os.path.exists(f), f"File {f} does not exist"
            data = np.load(f, allow_pickle=True)

            obj = data["arr_0"]

            if obj.ndim == 0:
                # Access the single element
                obj = obj.item()

            for k in obj.keys():
                if k not in predictions:
                    predictions[k] = {}
                else:
                    print(f"Skipping {k}: already exists in predictions")
                    continue

                predictions[k] = obj[k]
            print("Read predictions from {}".format(f))
    else:
        predictions = predict_many(args, opts_list)

    if args.include_additional is not None:
        for i, f in enumerate(args.include_additional):
            prod, file = f.split(":")
            assert os.path.exists(file), f"File {file} does not exist"

            print("Reading additional predictions for {} from {}".format(prod, file))
            data = np.load(file, allow_pickle=True)

            if "arr_0" in data.keys():
                data = data["arr_0"].item()

            predictions[prod] = data[prod]

    if "exim" in predictions.keys():
        # exim is missing analysis time data, add it
        predictions["exim"] = amend_exim_with_analysis_time(
            predictions["gt"], predictions["exim"]
        )

    if len(predictions.keys()) > 2:  # more than gt and single model
        predictions = intersect_and_filter(args, predictions)

    for k in predictions.keys():
        if k == "gt":
            print(f"Found {len(predictions[k]['data'])} ground truth images")
        else:
            print(f"Found {len(predictions[k]['data'])} forecasts for {k}")

    plot_timeseries(args, predictions)
    produce_scores(args, predictions)

    if args.plot_dir is None:
        plt.show()
