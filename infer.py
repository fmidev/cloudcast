from tensorflow.keras.models import load_model
from model import *
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from fileutils import *
from preprocess import *
from plotutils import *
from generators import *
from verifutils import *
from skimage.metrics import structural_similarity as ssim

PRED_STEP = timedelta(minutes=15)
DSS = {}

def parse_time(timestr):
    masks = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']
    for m in masks:
        try:
            return datetime.datetime.strptime(timestr, m)
        except ValueError as e:
            pass

    return None


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=False)
    parser.add_argument("--stop_date", action='store', type=str, required=False)
    parser.add_argument("--single_time", action='store', type=str, required=False)
    parser.add_argument("--label", action='store', nargs='+', type=str, required=True)
    parser.add_argument("--save_grib", action='store_true', default=False)
    parser.add_argument("--disable_plot", action='store_true', default=False)
    parser.add_argument("--prediction_len", action='store', type=int, default=12)
    parser.add_argument("--exclude_analysistime", action='store_true', default=False, help='exclude analysistime from data')
    parser.add_argument("--include_additional", action='store', nargs='+', default=[])
    parser.add_argument("--top", action='store', type=int, default=-1, help='out of all models select the top n that perform best')

    args = parser.parse_args()

    if (args.start_date is None and args.stop_date is None) and args.single_time is None:
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

    for lbl in args.label:
        if lbl.find('*') != -1:
            lbls = [os.path.basename(x) for x in glob.glob(f"models/{lbl}")]
            normalized_labels.extend(lbls)

        else:
            normalized_labels.append(lbl)

    return normalized_labels

def infer_many(m, orig, num_predictions, **kwargs):
    datetime_weights = kwargs.get('datetime_weights', None)
    environment_weights = kwargs.get('environment_weights', None)
    leadtime_conditioning = kwargs.get('leadtime_conditioning', None)

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


def predict_many(args):
    all_pred = {}

    for lbl in args.label:
        elem = copy.deepcopy(args)
        elem.label = lbl

        predictions = predict(elem)

        for i,k in enumerate(predictions.keys()):
            if not k in all_pred.keys():
                all_pred[k] = predictions[k]

    return all_pred

def predict(args):
    global DSS

    args.model, args.loss_function, args.n_channels, args.include_datetime, args.include_environment_data, args.leadtime_conditioning, args.preprocess = args.label.split('-')
    args.n_channels = int(args.n_channels)
    args.include_datetime = eval(args.include_datetime)
    args.include_environment_data = eval(args.include_environment_data)
    args.leadtime_conditioning = eval(args.leadtime_conditioning)

    model_file = 'models/{}'.format(get_model_name(args))
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    try:
        dss = DSS[args.preprocess]
    except KeyError as e:
        DSS[args.preprocess] = {}
        dss = DSS[args.preprocess]

    time_gen = TimeseriesGenerator(args.start_date, args.stop_date, args.n_channels, args.prediction_len, step=PRED_STEP)
    environment_weights = None

    nwp = []
    climatology = False

    for k in args.include_additional:
        if k == 'climatology':
            climatology=True
        else:
            nwp.append(k)

    if not 'nwcsaf' in dss:
        dss['nwcsaf'] = DataSeries("nwcsaf", args.preprocess)

    predictions = {
        args.label : { 'time' : [], 'data' : [] },
        'gt' : { 'time' : [], 'data' : [] },
    }

    if climatology:
        predictions['climatology'] = { 'time' : [], 'data' : [] }

    for k in nwp:
        if not k in dss:
            dss[k] = DataSeries(k, args.preprocess)
        predictions[k] = { 'time' : [], 'data' : [] }

    def diff(a, b):
        b = set(b)
        return [i for i in a if i not in b]

    for times in time_gen:
        history = times[:args.n_channels]
        leadtimes = times[args.n_channels:]

        print("Using history {} to predict {}".format(
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), history)),
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), leadtimes))
        ))

        datas = {}
        if args.disable_plot:
            datas['nwcsaf'] = dss['nwcsaf'].read_data(history)
            initial = np.copy(datas['nwcsaf'][-1])

        else:
            for k in dss:
                analysis_time = None
                _leadtimes = leadtimes.copy()
                if not args.exclude_analysistime:
                    _leadtimes = [times[args.n_channels-1]] + leadtimes
                if k in ['meps','mnwc']:
                    analysis_time = times[args.n_channels].replace(minute=0)
                    datas[k] = dss[k].read_data(_leadtimes, analysis_time)
                elif k == 'nwcsaf':
                    datas[k] = dss[k].read_data(times)

            initial = np.copy(datas['nwcsaf'][args.n_channels - 1])

            if climatology:
                clim = generate_clim_values((len(leadtimes),) + get_img_size(args.preprocess), int(leadtimes[0].strftime("%m")))

        if np.isnan(datas['nwcsaf']).any():
            print("Seed contains missing values, skipping")
            continue

        if args.disable_plot:
            new_times = diff(history, predictions['gt']['time'])
        else:
            new_times = diff(times, predictions['gt']['time'])

        for t in new_times:
            i = times.index(t)
            predictions['gt']['time'].append(t)
            predictions['gt']['data'].append(datas['nwcsaf'][i])

#        if args.disable_plot is True:
#            datas['nwcsaf'] = datas['nwcsaf'][args.n_channels:]

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
                lt.append(create_squeezed_leadtime_conditioning(get_img_size(args.preprocess), args.leadtime_conditioning, i))
            lt = np.squeeze(np.asarray(lt), axis=1)

        if args.model == "unet":
            cc = infer_many(m, datas['nwcsaf'][:args.n_channels], args.prediction_len, datetime_weights=datetime_weights, environment_weights=environment_weights, leadtime_conditioning=lt)
        else:
            hist = datas['nwcsaf'][:args.n_channels]

            if args.include_environment_data:
                dt0 = np.tile(environment_weights[0], 6)
                dt0 = np.swapaxes(np.expand_dims(dt0, axis=0), 0, 3)
                dt1 = np.tile(environment_weights[1], 6)
                dt1 = np.swapaxes(np.expand_dims(dt1, axis=0), 0, 3)
                hist = np.concatenate((hist, dt0, dt1), axis=-1)
                assert(np.max(dt0) <= 1 and np.max(dt1) <= 1)
            cc = predict_from_series(m, hist, args.prediction_len)

        if args.exclude_analysistime is False:
            cc = np.concatenate((np.expand_dims(datas['nwcsaf'][args.n_channels-1], axis=0), cc), axis=0)
            leadtimes = [history[-1]] + leadtimes

        assert(cc.shape[0] == len(leadtimes))
        predictions[args.label]['time'].append(leadtimes)
        predictions[args.label]['data'].append(cc)

        if args.disable_plot:
            continue

        for k in nwp:
            predictions[k]['time'].append(leadtimes)
            predictions[k]['data'].append(datas[k])

        if climatology:
            predictions['climatology']['time'].append(leadtimes)
            predictions['climatology']['data'].append(clim)

        future_data = datas['nwcsaf'][args.n_channels:]

        if args.exclude_analysistime is False:
            future_data = datas['nwcsaf'][args.n_channels-1:]


    return predictions



def copy_range(gt, start, stop):
    a = gt['time'].index(start)
    b = gt['time'].index(stop)
    return np.asarray(gt['data'][a:b+1]) # need inclusive end



def sort_errors(errors, best_first=True):
    assert(best_first)

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
            for i,j in enumerate(labels):
                if j not in keep:
                    labels[i] = k
                    break
        labels.reverse()

    f_predictions = {}
    f_errors = {}

    for l in labels:
        predkey = l
        if l == 'persistence':
            predkey = 'gt'
        f_predictions[predkey] = predictions[predkey]
        f_errors[l] = errors[l]

    return f_predictions, f_errors


def plot_timeseries(args, predictions):

    while True:
        first = list(predictions.keys())[np.random.randint(len(predictions))]
        if first != 'gt':
            break

    if len(predictions) < 8:
        labels = list(predictions.keys())
        try:
            labels.remove('climatology') # not plotting this in stamp plot
        except ValueError as e:
            pass

        labels.sort()
        idx = np.random.randint(len(predictions[first]['data']))

        data = []
        times = predictions[first]['time'][idx]

        for l in labels:
            if l == 'gt':
                data.append(copy_range(predictions['gt'], times[0], times[-1]))
                continue
            data.append(predictions[l]['data'][idx])

        plot_stamps(data, labels, title='Prediction for t0={}'.format(times[0]), initial_data=None, start_from_zero=(not args.exclude_analysistime))
    else:
        print("Too many predictions ({}) for timeseries plot".format(len(predictions)))


def save_gribs(args, predictions):

    for label in args.label:
        cc = predictions[label]
        alltimes = cc['time']
        alldata = cc['data']

        for data,times in zip(alldata, alltimes):
            assert(len(times) == len(data))

            analysistime = times[0]

            if args.exclude_analysistime:
                analysistime = analysistime - PRED_STEP
            for d,t in zip(data, times):
                leadtime = int((t - analysistime).total_seconds()/60)
                filename = '/tmp/{}/{}+{:03d}m.grib2'.format(label, analysistime.strftime('%Y%m%d%H%M%S'), leadtime)
                save_grib(d, filename, analysistime, t)


if __name__ == "__main__":
    args = parse_command_line()

    args.label = normalize_label(args.label)
    predictions = predict_many(args)
#    predictions, errors = filter_top_n(predictions, errors, args.top, keep=['persistence'] + args.include_additional)

    if args.save_grib:
        save_gribs(args, predictions)

    if not args.disable_plot:
        plot_timeseries(args, predictions)
        produce_scores(args, predictions)

        plt.show()
