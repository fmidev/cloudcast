from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import numpy as np
import argparse
from datetime import datetime, timedelta
from base.fileutils import *
from base.preprocess import *
from base.generators import *
from base.postprocess import *
from base.opts import CloudCastOptions
from base.dataseries import LazyDataSeries

PRED_STEP = timedelta(minutes=15)


def parse_command_line():
    def valid_time(x):
        try:
            return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)

    def output_size(x):
        try:
            return tuple(map(int, x.split("x")))
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_time", action="store", type=valid_time, required=True
    )
    parser.add_argument("--label", action="store", type=str, required=True)
    parser.add_argument("--directory", action="store", default="/tmp")
    parser.add_argument("--prediction_len", action="store", type=int, default=12)
    parser.add_argument(
        "--output_size",
        action="store",
        type=output_size,
        default=None,
        help="downsampled size hxw",
    )

    args = parser.parse_args()
    args.onehot_encoding = False

    return args


def predict(args):
    opts = CloudCastOptions(label=args.label)

    model_file = "models/{}".format(opts.get_label())
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    filenames = list(range(0, opts.n_channels * 15, 15))
    filenames.reverse()
    filenames = list(
        map(
            lambda x: get_filename(args.analysis_time - datetime.timedelta(minutes=x)),
            filenames,
        )
    )

    lds = LazyDataSeries(
        opts=opts,
        filenames=filenames,
        operating_mode="INFER",
        **vars(args),
        enable_debug=True,
        enable_cache=True,
    )

    d = lds.get_dataset()

    forecast = []
    times = []

    for t in tfds.as_numpy(d):
        x = t[0]
        y = t[1]
        xy_times = np.squeeze(t[2])
        print(
            "Using {} to predict {}".format(
                list(map(lambda x: x.decode("utf8"), xy_times[:-1])),
                xy_times[-1].decode("utf8"),
            )
        )

        if np.isnan(x).any():
            print("Seed contains missing values, aborting")
            return None, None

        if len(forecast) == 0:
            # Add leadtime=0 ie. nwcsaf data ie. analysis time
            times.append(
                datetime.datetime.strptime(xy_times[-2].decode("utf8"), "%Y%m%dT%H%M%S")
            )
            forecast.append(x[..., lds.n_channels - 1])
            forecast[-1] = np.expand_dims(np.squeeze(forecast[-1]), axis=-1)

        prediction = m.predict(x)
        prediction = np.squeeze(prediction, axis=0)

        forecast.append(prediction)
        times.append(
            datetime.datetime.strptime(xy_times[-1].decode("utf8"), "%Y%m%dT%H%M%S")
        )

    if args.output_size is not None:
        _forecast = []
        for d in forecast:
            _d = downscale(np.squeeze(d), args.output_size)
            _d = np.expand_dims(_d, axis=-1)
            _forecast.append(_d)
        forecast = _forecast

    return times, forecast


def save_gribs(args, times, data):
    if times is None:
        return

    analysistime = times[0]

    for d, t in zip(data, times):
        leadtime = int((t - analysistime).total_seconds() / 60)
        filename = "{}/{}+{:03d}m.grib2".format(
            args.directory, analysistime.strftime("%Y%m%d%H%M%S"), leadtime
        )

        save_grib(d, filename, analysistime, t)


if __name__ == "__main__":
    args = parse_command_line()

    times, data = predict(args)
    save_gribs(args, times, data)
