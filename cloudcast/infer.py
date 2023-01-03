from tensorflow.keras.models import load_model
import numpy as np
import argparse
from datetime import datetime, timedelta
from base.fileutils import *
from base.preprocess import *
from base.generators import *
from base.postprocess import *
from base.opts import CloudCastOptions

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


def infer_many(m, orig, num_predictions, **kwargs):
    datetime_weights = kwargs.get("datetime_weights", None)
    topography_weights = kwargs.get("topography_weights", None)
    terrain_type_weights = kwargs.get("terrain_type_weights", None)
    leadtime_conditioning = kwargs.get("leadtime_conditioning", None)
    onehot_encoding = kwargs.get("onehot_encoding", False)

    predictions = []
    hist_len = len(orig)
    orig_sq = np.squeeze(np.moveaxis(orig, 0, 3), -2)

    def create_hist(predictions):
        if len(predictions) == 0:
            return np.squeeze(np.moveaxis(orig, 0, 3), -2)
        elif len(predictions) >= hist_len:
            return np.squeeze(
                np.moveaxis(np.asarray(predictions[-hist_len:]), 0, 3), -2
            )

        hist_a = orig[: hist_len - len(predictions)]
        hist_b = np.asarray(predictions[-len(predictions) :])

        seed = np.squeeze(
            np.moveaxis(np.concatenate((hist_a, hist_b), axis=0), 0, 3), -2
        )

        return seed

    def append_auxiliary_weights(
        data, datetime_weights, topography_weights, terrain_type_weights, num_prediction
    ):
        if leadtime_conditioning is not None:
            if onehot_encoding:
                lts = np.squeeze(
                    np.swapaxes(leadtime_conditioning[num_prediction], 0, -1)
                )
                data = np.concatenate((data, lts), axis=-1)
            else:
                data = np.concatenate(
                    (data, leadtime_conditioning[num_prediction]), axis=-1
                )

        if datetime_weights is not None:
            data = np.concatenate(
                (
                    data,
                    datetime_weights[hist_len - 1][0],
                    datetime_weights[hist_len - 1][1],
                ),
                axis=-1,
            )

        if topography_weights is not None:
            data = np.concatenate((data, topography_weights), axis=-1)

        if terrain_type_weights is not None:
            data = np.concatenate((data, terrain_type_weights), axis=-1)

        return data

    data = orig_sq

    for i in range(num_predictions):

        if leadtime_conditioning is None:
            # autoregression
            data = create_hist(predictions)

        alldata = append_auxiliary_weights(
            data, datetime_weights, topography_weights, terrain_type_weights, i
        )
        pred = infer(m, alldata)
        predictions.append(pred)

    return np.asarray(predictions)


def infer(m, img):
    img = np.expand_dims(img, axis=0)
    prediction = m.predict(img)
    pred = np.squeeze(prediction, axis=0)
    return pred


def predict_from_series(m, dataseries, num):
    pred = m.predict(np.expand_dims(dataseries, axis=0))
    pred = np.squeeze(pred, axis=0)

    if pred.shape[0] == num:
        return pred

    moar = m.predict(np.expand_dims(pred, axis=0))
    moar = np.squeeze(moar, axis=0)

    comb = np.concatenate((pred, moar), axis=0)
    return comb[:num]


def predict(args):

    opts = CloudCastOptions(label=args.label)

    model_file = "models/{}".format(opts.get_label())
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    dss = DataSeries("nwcsaf", opts.preprocess, fill_gaps_max=1)

    time_gen = iter(
        TimeseriesGenerator(
            args.analysis_time,
            args.analysis_time,
            opts.n_channels,
            args.prediction_len,
            step=PRED_STEP,
        )
    )
    topography_weights = None
    terrain_type_weights = None

    predictions = {"time": [], "data": []}

    def diff(a, b):
        b = set(b)
        return [i for i in a if i not in b]

    times = next(time_gen)
    history = times[: opts.n_channels]
    leadtimes = times[opts.n_channels :]

    print(
        "Using history {} to predict {}".format(
            list(map(lambda x: "{}".format(x.strftime("%H:%M")), history)),
            list(map(lambda x: "{}".format(x.strftime("%H:%M")), leadtimes)),
        )
    )

    datas = dss.read_data(history)

    if np.isnan(datas).any():
        print("Seed contains missing values, aborting")
        return None

    datetime_weights = None
    lt = None

    if opts.include_datetime:
        datetime_weights = list(
            map(lambda x: create_datetime(x, get_img_size(opts.preprocess)), history)
        )

    if opts.include_topography and topography_weights is None:
        topography_weights = create_topography_data(
            opts.preprocess, opts.model == "convlstm"
        )

    if opts.include_terrain_type and terrain_type_weights is None:
        terrain_typey_weights = create_terrain_type_data(
            opts.preprocess, opts.model == "convlstm"
        )

    if opts.leadtime_conditioning:
        assert args.prediction_len <= opts.leadtime_conditioning
        lt = []
        for i in range(args.prediction_len):
            if opts.onehot_encoding is False:
                lt.append(
                    create_squeezed_leadtime_conditioning(
                        get_img_size(opts.preprocess), opts.leadtime_conditioning, i
                    )
                )
            else:
                lt.append(
                    create_onehot_leadtime_conditioning(
                        get_img_size(opts.preprocess), opts.leadtime_conditioning, i
                    )
                )

        if opts.onehot_encoding is False:
            lt = np.squeeze(np.asarray(lt), axis=1)

    if opts.model == "unet":
        cc = infer_many(
            m,
            datas[: opts.n_channels],
            args.prediction_len,
            datetime_weights=datetime_weights,
            topography_weights=topography_weights,
            terrain_type_weights=terrain_type_weights,
            leadtime_conditioning=lt,
            onehot_encoding=opts.onehot_encoding,
        )
    else:
        hist = datas[: opts.n_channels]

        if opts.include_topography_data:
            topo = np.tile(topography_weights, 6)
            topo = np.swapaxes(np.expand_dims(topo, axis=0), 0, 3)
            hist = np.concatenate((hist, topo), axis=-1)

        if opts.include_terrain_type_data:
            terr = np.tile(terrain_type_weights, 6)
            terr = np.swapaxes(np.expand_dims(terr, axis=0), 0, 3)
            hist = np.concatenate((hist, terr), axis=-1)

        cc = predict_from_series(m, hist, args.prediction_len)

    cc = np.concatenate(
        (np.expand_dims(datas[opts.n_channels - 1], axis=0), cc), axis=0
    )
    leadtimes = [history[-1]] + leadtimes

    if args.output_size is not None:
        ccn = []
        for i, _cc in enumerate(cc):
            _ccn = downscale(np.squeeze(_cc), args.output_size)
            _ccn = np.expand_dims(_ccn, axis=-1)
            ccn.append(_ccn)
        cc = np.asarray(ccn)

    assert cc.shape[0] == len(leadtimes)
    predictions["time"].append(leadtimes)
    predictions["data"].append(cc)

    return predictions


def save_gribs(args, predictions):
    if predictions is None:
        return

    alltimes = predictions["time"]
    alldata = predictions["data"]

    for data, times in zip(alldata, alltimes):
        assert len(times) == len(data)

        analysistime = times[0]

        for d, t in zip(data, times):
            leadtime = int((t - analysistime).total_seconds() / 60)
            filename = "{}/{}+{:03d}m.grib2".format(
                args.directory, analysistime.strftime("%Y%m%d%H%M%S"), leadtime
            )
            save_grib(d, filename, analysistime, t)


if __name__ == "__main__":
    args = parse_command_line()

    predictions = predict(args)
    save_gribs(args, predictions)
