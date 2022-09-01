from tensorflow.keras.models import load_model
import numpy as np
import argparse
from datetime import datetime, timedelta
from base.fileutils import *
from base.preprocess import *
from base.generators import *
from base.postprocess import *

PRED_STEP = timedelta(minutes=15)

def parse_command_line():
    def valid_time(x):
        try:
            return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)

    def output_size(x):
        try:
            return tuple(map(int, x.split('x')))
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)

    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_time", action='store', type=valid_time, required=True)
    parser.add_argument("--label", action='store', type=str, required=True)
    parser.add_argument("--directory", action='store', default='/tmp')
    parser.add_argument("--prediction_len", action='store', type=int, default=12)
    parser.add_argument("--output_size", action='store', type=output_size, default=None, help='downsampled size hxw')

    args = parser.parse_args()
    args.onehot_conditioning = False

    return args



def infer_many(m, orig, num_predictions, **kwargs):
    datetime_weights = kwargs.get('datetime_weights', None)
    environment_weights = kwargs.get('environment_weights', None)
    leadtime_conditioning = kwargs.get('leadtime_conditioning', None)
    onehot_conditioning = kwargs.get('onehot_conditioning', False)

    predictions = []
    hist_len = len(orig)
    orig_sq = np.squeeze(np.moveaxis(orig, 0, 3), -2)

    def create_hist(predictions):
        if len(predictions) == 0:
            return np.squeeze(np.moveaxis(orig, 0, 3), -2)
        elif len(predictions) >= hist_len:
             return np.squeeze(np.moveaxis(np.asarray(predictions[-hist_len:]), 0, 3), -2)

        hist_a = orig[:hist_len-len(predictions)]
        hist_b = np.asarray(predictions[-len(predictions):])

        seed = np.squeeze(np.moveaxis(np.concatenate((hist_a, hist_b), axis=0), 0, 3), -2)

        return seed

    def append_auxiliary_weights(data, datetime_weights, environment_weights, num_prediction):
        if leadtime_conditioning is not None:
            if onehot_conditioning:
                lts = np.squeeze(np.swapaxes(leadtime_conditioning[num_prediction], 0, -1))
                data = np.concatenate((data, lts), axis=-1)
            else:
                data = np.concatenate((data, leadtime_conditioning[num_prediction]), axis=-1)

        if datetime_weights is not None:
            data = np.concatenate((data, datetime_weights[hist_len-1][0], datetime_weights[hist_len-1][1]), axis=-1)

        if environment_weights is not None:
            data = np.concatenate((data, environment_weights[0], environment_weights[1]), axis=-1)

        return data

    data = orig_sq

    for i in range(num_predictions):

        if leadtime_conditioning is None:
            # autoregression
            data = create_hist(predictions)

        alldata = append_auxiliary_weights(data, datetime_weights, environment_weights, i)
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

    args.model, args.loss_function, args.n_channels, args.include_datetime, args.include_environment_data, args.leadtime_conditioning, args.preprocess = args.label.split('-')
    args.n_channels = int(args.n_channels)
    args.include_datetime = eval(args.include_datetime)
    args.include_environment_data = eval(args.include_environment_data)
    args.leadtime_conditioning = eval(args.leadtime_conditioning)

    model_file = 'models/{}'.format(get_model_name(args))
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    dss = DataSeries("nwcsaf", args.preprocess, fill_gaps_max=1)

    time_gen = iter(TimeseriesGenerator(args.analysis_time, args.analysis_time, args.n_channels, args.prediction_len, step=PRED_STEP))
    environment_weights = None

    predictions = { 'time' : [], 'data' : [] }

    def diff(a, b):
        b = set(b)
        return [i for i in a if i not in b]

    times = next(time_gen)
    history = times[:args.n_channels]
    leadtimes = times[args.n_channels:]

    print("Using history {} to predict {}".format(
        list(map(lambda x: '{}'.format(x.strftime('%H:%M')), history)),
        list(map(lambda x: '{}'.format(x.strftime('%H:%M')), leadtimes))
    ))

    datas = dss.read_data(history)

    if np.isnan(datas).any():
        print("Seed contains missing values, aborting")
        return None

    datetime_weights = None
    lt = None

    if args.include_datetime:
        datetime_weights = list(map(lambda x: create_datetime(x, get_img_size(args.preprocess)), history))

    if args.include_environment_data and environment_weights is None:
        environment_weights = create_environment_data(args.preprocess, args.model == 'convlstm')

    if args.leadtime_conditioning:
        assert(args.prediction_len <= args.leadtime_conditioning)
        lt = []
        for i in range(args.prediction_len):
            if args.onehot_conditioning is False:
                lt.append(create_squeezed_leadtime_conditioning(get_img_size(args.preprocess), args.leadtime_conditioning, i))
            else:
                lt.append(create_onehot_leadtime_conditioning(get_img_size(args.preprocess), args.leadtime_conditioning, i))

        if args.onehot_conditioning is False:
            lt = np.squeeze(np.asarray(lt), axis=1)

    if args.model == "unet":
        cc = infer_many(m, datas[:args.n_channels], args.prediction_len, datetime_weights=datetime_weights, environment_weights=environment_weights, leadtime_conditioning=lt, onehot_conditioning=args.onehot_conditioning)
    else:
        hist = datas[:args.n_channels]

        if args.include_environment_data:
            dt0 = np.tile(environment_weights[0], 6)
            dt0 = np.swapaxes(np.expand_dims(dt0, axis=0), 0, 3)
            dt1 = np.tile(environment_weights[1], 6)
            dt1 = np.swapaxes(np.expand_dims(dt1, axis=0), 0, 3)
            hist = np.concatenate((hist, dt0, dt1), axis=-1)
            assert(np.max(dt0) <= 1 and np.max(dt1) <= 1)
        cc = predict_from_series(m, hist, args.prediction_len)

    cc = np.concatenate((np.expand_dims(datas[args.n_channels-1], axis=0), cc), axis=0)
    leadtimes = [history[-1]] + leadtimes

    if args.output_size is not None:
        ccn = []
        for i,_cc in enumerate(cc):
            _ccn = downscale(np.squeeze(_cc), args.output_size)
            _ccn = np.expand_dims(_ccn, axis=-1)
            ccn.append(_ccn)
        cc = np.asarray(ccn)

    assert(cc.shape[0] == len(leadtimes))
    predictions['time'].append(leadtimes)
    predictions['data'].append(cc)

    return predictions


def save_gribs(args, predictions):
    if predictions is None:
        return

    alltimes = predictions['time']
    alldata = predictions['data']

    for data,times in zip(alldata, alltimes):
        assert(len(times) == len(data))

        analysistime = times[0]

        for d,t in zip(data, times):
            leadtime = int((t - analysistime).total_seconds()/60)
            filename = '{}/{}+{:03d}m.grib2'.format(args.directory, analysistime.strftime('%Y%m%d%H%M%S'), leadtime)
            save_grib(d, filename, analysistime, t)


if __name__ == "__main__":
    args = parse_command_line()

    predictions = predict(args)
    save_gribs(args, predictions)
