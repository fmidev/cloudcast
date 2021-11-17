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

PRED_STEP = timedelta(minutes=15)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=False)
    parser.add_argument("--stop_date", action='store', type=str, required=False)
    parser.add_argument("--single_time", action='store', type=str, required=False)
    parser.add_argument("--label", action='append', type=str, required=True)
    parser.add_argument("--save_grib", action='store_true', default=False)
    parser.add_argument("--disable_plot", action='store_true', default=False)
    parser.add_argument("--prediction_len", action='store', type=int, default=12)
    parser.add_argument("--exclude_analysistime", action='store_true', default=False)
    parser.add_argument("--include_climatology", action='store_true', default=False)
    args = parser.parse_args()

    if (args.start_date is None and args.stop_date is None) and args.single_time is None:
        print("One of: (start_date, stop_date), (single_time) must be given")
        sys.exit(1)

    if args.start_date is not None and args.stop_date is not None:
        args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')
    else:
        try:
            args.start_date = datetime.datetime.strptime(args.single_time, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            args.start_date = datetime.datetime.strptime(args.single_time, '%Y-%m-%dT%H:%M:%S')
        args.stop_date = args.start_date

    return args



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
    all_err = {}

    for lbl in args.label:
        elem = copy.deepcopy(args)
        elem.label = lbl

        predictions, errors = predict(elem)

        for i,k in enumerate(predictions.keys()):
            if not k in all_pred.keys():
                all_pred[k] = predictions[k]

        for i,k in enumerate(errors.keys()):
            if not k in all_err.keys():
                all_err[k] = errors[k]

    return all_pred, all_err

def predict(args):
    args.model, args.loss_function, args.n_channels, args.include_datetime, args.include_environment_data, args.leadtime_conditioning, args.preprocess = args.label.split('-')
    args.n_channels = int(args.n_channels)
    args.include_datetime = eval(args.include_datetime)
    args.include_environment_data = eval(args.include_environment_data)
    args.leadtime_conditioning = eval(args.leadtime_conditioning)

    model_file = 'models/{}'.format(get_model_name(args))
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    time_gen = TimeseriesGenerator(args.start_date, args.n_channels, args.prediction_len, step=PRED_STEP, stop_date=args.stop_date)

    mae_cc = []
    mae_prst = []
    mae_clim = []
    mae_mnwc = []
    mae_meps = []

    environment_weights = None

    gt_ds = DataSeries("nwcsaf", args.preprocess)
    mnwc_ds = DataSeries("mnwc", args.preprocess)
    meps_ds = DataSeries("meps", args.preprocess)

    predictions = {
        args.label : { 'time' : [], 'data' : [] },
        'mnwc' : { 'time' : [], 'data' : [] },
        'meps' : { 'time' : [], 'data' : [] },
        'gt' : { 'time' : [], 'data' : [] },
        'clim' : { 'time' : [], 'data' : [] }
    }

    for i in range(args.prediction_len):
        mae_cc.append([])
        mae_prst.append([])
        mae_mnwc.append([])
        mae_meps.append([])
        mae_clim.append([])

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

        if args.disable_plot:
            gt = gt_ds.read_data(history)
            initial = np.copy(gt[-1])

        else:
            gt = gt_ds.read_data(times)
            mnwc = mnwc_ds.read_data(leadtimes, times[args.n_channels].replace(minute=0))
            meps = meps_ds.read_data(leadtimes, times[args.n_channels].replace(minute=0))
            initial = np.copy(gt[args.n_channels - 1])

            if args.include_climatology:
                clim = generate_clim_values((len(leadtimes),) + get_img_size(args.preprocess))

        if np.isnan(gt).any():
            print("Seed contains missing values, skipping")
            continue

        if args.disable_plot:
            new_times = diff(history, predictions['gt']['time'])
        else:
            new_times = diff(times, predictions['gt']['time'])

        for t in new_times:
            i = times.index(t)
            predictions['gt']['time'].append(t)
            predictions['gt']['data'].append(gt[i])

        if args.disable_plot is False:
            gt = gt[args.n_channels:]

        datetime_weights = None
        lt = None

        if args.include_datetime:
            datetime_weights = list(map(lambda x: create_datetime(x, get_img_size(args.preprocess)), leadtimes))
        if args.include_environment_data and environment_weights is None:
            environment_weights = create_environment_data(args.preprocess)
        if args.leadtime_conditioning:
            assert(args.prediction_len <= args.leadtime_conditioning)
            lt = []
            for i in range(args.prediction_len):
                lt.append(create_squeezed_leadtime_conditioning(get_img_size(args.preprocess), args.leadtime_conditioning, i))
            lt = np.squeeze(np.asarray(lt), axis=1)

        if args.model == "unet":
            cc = infer_many(m, gt[:args.n_channels], args.prediction_len, datetime_weights=datetime_weights, environment_weights=environment_weights, leadtime_conditioning=lt)
        else:
            cc = predict_from_series(m, gt[:args.n_channels], args.prediction_len)

        if args.disable_plot and not args.exclude_analysistime:
            cc = np.concatenate((np.expand_dims(gt[-1], axis=0), cc), axis=0)
            leadtimes = [history[-1]] + leadtimes

        assert(cc.shape[0] == len(leadtimes))
        predictions[args.label]['time'].append(leadtimes)
        predictions[args.label]['data'].append(cc)

        if not args.disable_plot:
            predictions['mnwc']['time'].append(leadtimes)
            predictions['mnwc']['data'].append(mnwc)
            predictions['meps']['time'].append(leadtimes)
            predictions['meps']['data'].append(meps)

            if args.include_climatology:
                predictions['clim']['time'].append(leadtimes)
                predictions['clim']['data'].append(clim)

        for i,t in enumerate(gt):
            if np.isnan(t).any():
                continue

            if not args.disable_plot and not np.isnan(mnwc[i]).any():
                mae_mnwc[i].append(mean_absolute_error(t.flatten(), mnwc[i].flatten()))
            if not args.disable_plot and not np.isnan(meps[i]).any():
                mae_meps[i].append(mean_absolute_error(t.flatten(), meps[i].flatten()))

            mae_prst[i].append(mean_absolute_error(t.flatten(), initial.flatten()))
            mae_cc[i].append(mean_absolute_error(t.flatten(), cc[i].flatten()))

            if args.include_climatology:
                mae_clim[i].append(mean_absolute_error(t.flatten(), clim[i].flatten()))

    return predictions, {'prst' : mae_prst, args.label : mae_cc, 'mnwc' : mae_mnwc, 'meps' : mae_meps, 'clim' : mae_clim }



def copy_range(gt, start, stop):
    a = gt['time'].index(start)
    b = gt['time'].index(stop)
    return np.asarray(gt['data'][a:b+1]) # need inclusive end


def calculate_errors(models, predictions):
    errors = {}

    for m in models:
        errors[m] = [[]]*args.prediction_len

    for m in models:
        for i, pred in enumerate(predictions[m]['data']):
            times = predictions[m]['time'][i]
            gt = copy_range(times[0], times[-1])


#    for elem in predictions['gt']: 
#        time = elem['time']
#        data = elem['data']

        
        #for k in predictions.keys():



def plot_results(args, predictions, errors):

    labels = [ 'ground truth', 'mnwc', 'meps' ]
    if args.include_climatology:
        labels.append('clim')

    labels.extend(args.label)

    idx = np.random.randint(len(predictions['mnwc']['data']))

    data = []

    for l in labels[1:]:
        data.append(predictions[l]['data'][idx])

    times = predictions[args.label[0]]['time'][idx]
    gt = copy_range(predictions['gt'], times[0], times[-1])

    data = [gt] + data #, pred_mnwc, pred_meps, pred_clim]

    plot_timeseries(data, labels, title='Prediction for t0={}'.format(times[0]), initial_data=None) #predictions['gt']['data'][0])

    #######################

    labels = [ 'persistence', 'mnwc', 'meps' ]
    if args.include_climatology:
        labels.append('clim')

    labels.extend(args.label)

    data = []

    for l in labels[1:]:
        data.append(errors[l])

    data = [errors['prst']] + data

    for i,m in enumerate(data):
        for j,lt in enumerate(m):
            data[i][j] = np.mean(data[i][j])

    plot_mae(data, labels, title='MAE over {} predictions'.format(len(predictions['mnwc']['data'])))

    plt.pause(0.001)
    input("Press [enter] to stop")



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

    predictions, errors = predict_many(args)

    if not args.disable_plot:
        plot_results(args, predictions, errors)

    if args.save_grib:
        save_gribs(args, predictions)
