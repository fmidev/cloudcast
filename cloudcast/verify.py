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
    parser.add_argument("--single_time", action="store", type=str, required=False)
    parser.add_argument("--label", action="store", nargs="+", type=str, required=True)
    parser.add_argument("--prediction_len", action="store", type=int, default=12)
    parser.add_argument("--include_additional", action="store", nargs="+", default=[])
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

    args = parser.parse_args()

    if (
        args.start_date is None and args.stop_date is None
    ) and args.single_time is None:
        print("One of: (start_date, stop_date), (single_time) must be given")
        sys.exit(1)

    if args.start_date is not None and args.stop_date is not None:
        args.start_date = parse_time(args.start_date)
        args.stop_date = parse_time(args.stop_date)
    else:
        args.start_date = parse_time(args.single_time)
        args.stop_date = args.start_date

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

    assert opts.leadtime_conditioning == 12

    for t in tfds.as_numpy(d):
        x = t[0]
        y = t[1]
        xy_times = np.squeeze(t[2])
        xy_times = list(map(lambda x: x.decode("utf8"), xy_times))

        print("Using {} to predict {}".format(xy_times[:-1], xy_times[-1]))

        if len(forecast) > 0 and len(forecast) % 13 == 0:
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

        prediction = m.predict(x, verbose=0)
        prediction = np.squeeze(prediction, axis=0)
        prediction_time = datetime.datetime.strptime(xy_times[-1], "%Y%m%dT%H%M%S")
        forecast.append(prediction)
        times.append(prediction_time)

        if prediction_time not in predictions["gt"]["time"]:
            predictions["gt"]["time"].append(prediction_time)
            predictions["gt"]["data"].append(np.squeeze(y, axis=0))

    if len(forecast) == 13:
        predictions[opts.get_label()]["time"].append(times.copy())
        predictions[opts.get_label()]["data"].append(np.array(forecast))

    n_forecast = len(predictions[opts.get_label()]["data"])

    print("Read {} forecasts in {:.1f} sec".format(n_forecast, time.time() - start))

    if n_forecast == 0:
        sys.exit(1)

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

    for l in labels:
        if l == "gt":
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
        plot_dir=args.plot_dir,
    )


def intersection(opts_list, predictions):
    # take intersection of predicted data so that we get same amount of forecasts
    # for same times
    # sometimes this might differ if for example history is missing and one model
    # needs more history than another

    labels = list(map(lambda x: x.get_label(), opts_list))

    def _intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    utimes = None

    for i in range(len(labels) - 1):
        label = labels[i]
        next_label = labels[i + 1]
        if utimes == None:
            utimes = _intersection(
                predictions[label]["time"], predictions[next_label]["time"]
            )
        else:
            utimes = _intersection(utimes, predictions[next_label]["time"])

    ret = {}

    for label in labels:
        ret[label] = {"time": utimes, "data": []}

        for utime in utimes:
            idx = predictions[label]["time"].index(utime)
            ret[label]["data"].append(predictions[label]["data"][idx])

    # copy other
    for label in predictions:
        if label not in labels:
            ret[label] = predictions[label]

    return ret


if __name__ == "__main__":
    args = parse_command_line()

    labels = normalize_label(args.label)
    opts_list = []
    for l in labels:
        opts_list.append(CloudCastOptions(label=l))
        assert opts_list[-1].onehot_encoding is False

    predictions = predict_many(args, opts_list)
    if args.start_date != args.stop_date and len(labels) > 1:
        predictions = intersection(opts_list, predictions)

    #    predictions, errors = filter_top_n(predictions, errors, args.top, keep=['persistence'] + args.include_additional)

    plot_timeseries(args, predictions)
    produce_scores(args, predictions)

    if args.plot_dir is None:
        plt.show()
