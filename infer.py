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

PRED_LEN = 12
PRED_STEP = timedelta(minutes=15)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=True)
    parser.add_argument("--stop_date", action='store', type=str, required=True)
    parser.add_argument("--label", action='append', type=str, required=True)

    args = parser.parse_args()

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args



def infer_many(m, orig, num_predictions, datetime_weights=None, environment_weights=None):
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

    def append_auxiliary_weights(data, datetime_weights, environment_weights):
        if datetime_weights is not None:
            data = np.concatenate((data, datetime_weights[hist_len-1][0], datetime_weights[hist_len-1][1]), axis=-1)

        if environment_weights is not None:
            data = np.concatenate((data, environment_weights[0], environment_weights[1]), axis=-1)

        return data

    for i in range(num_predictions):
        data = create_hist(predictions)
        data = append_auxiliary_weights(data, datetime_weights, environment_weights)

        pred = infer(m,data)
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
    args.model, args.loss_function, args.n_channels, args.include_datetime, args.include_environment_data, args.preprocess = args.label.split('-')
    args.n_channels = int(args.n_channels)
    args.include_datetime = eval(args.include_datetime)
    args.include_environment_data = eval(args.include_environment_data)

    model_file = 'models/{}'.format(get_model_name(args))
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    time_gen = TimeseriesGenerator(args.start_date - args.n_channels * PRED_STEP, PRED_LEN + args.n_channels, step=PRED_STEP, stop_date=args.stop_date)

    mae_cc = []
    mae_prst = []
    mae_mnwc = []

    environment_weights = None

    gt_ds = DataSeries("nwcsaf", args.preprocess)
    mnwc_ds = DataSeries("mnwc", args.preprocess)

    predictions = {
        args.label : { 'time' : [], 'data' : [] },
        'mnwc' : { 'time' : [], 'data' : [] },
        'gt' : { 'time' : [], 'data' : [] }
    }

    for i in range(PRED_LEN):
        mae_cc.append([])
        mae_prst.append([])
        mae_mnwc.append([])

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

        gt = gt_ds.read_data(times)

        mnwc = mnwc_ds.read_data(leadtimes, times[args.n_channels].replace(minute=0))
        initial = np.copy(gt[args.n_channels - 1])

        if np.isnan(gt).any():
            print("Seed contains missing values, skipping")
            continue

        new_times = diff(times, predictions['gt']['time'])
        for t in new_times:
            i = times.index(t)
            predictions['gt']['time'].append(t)
            predictions['gt']['data'].append(gt[i])


        gt = gt[args.n_channels:]

        datetime_weights = None

        if args.include_datetime:
            datetime_weights = list(map(lambda x: create_datetime(x, get_img_size(args.preprocess)), leadtimes))
        if args.include_environment_data and environment_weights is None:
            environment_weights = create_environment_data(args.preprocess)

        if args.model == "unet":
            cc = infer_many(m, gt[:args.n_channels], PRED_LEN, datetime_weights, environment_weights)
        else:
            cc = predict_from_series(m, gt[:args.n_channels], PRED_LEN)

        predictions[args.label]['time'].append(leadtimes)
        predictions[args.label]['data'].append(cc)
        predictions['mnwc']['time'].append(leadtimes)
        predictions['mnwc']['data'].append(mnwc)

        for i,t in enumerate(gt):
            if np.isnan(t).any():
                continue

            if not np.isnan(mnwc).any():
                mae_mnwc[i].append(mean_absolute_error(t.flatten(), mnwc[i].flatten()))

            mae_prst[i].append(mean_absolute_error(t.flatten(), initial.flatten()))
            mae_cc[i].append(mean_absolute_error(t.flatten(), cc[i].flatten()))

        break

    return predictions, {'prst' : mae_prst, args.label : mae_cc, 'mnwc' : mae_mnwc }



def copy_range(gt, start, stop):
    a = gt['time'].index(start)
    b = gt['time'].index(stop)
    return np.asarray(gt['data'][a:b+1]) # need inclusive end


def calculate_errors(models, predictions):
    errors = {}

    for m in models:
        errors[m] = [[]]*PRED_LEN

    for m in models:
        for i, pred in enumerate(predictions[m]['data']):
            times = predictions[m]['time'][i]
            gt = copy_range(times[0], times[-1])


#    for elem in predictions['gt']: 
#        time = elem['time']
#        data = elem['data']

        
        #for k in predictions.keys():



def plot_results(args, predictions, errors):

    labels = [ 'ground truth', 'mnwc' ]
    labels.extend(args.label)

    idx = np.random.randint(len(predictions['mnwc']['data']))
  
    pred_mnwc = predictions['mnwc']['data'][idx]
    times = predictions[args.label[0]]['time'][idx]
    gt = copy_range(predictions['gt'], times[0], times[-1])

    data = [gt, pred_mnwc]

    for l in args.label:
        data.append(predictions[l]['data'][idx])

    plot_timeseries(data, labels, title='Prediction for t0={}'.format(times[0]))

    #######################

    labels = [ 'persistence', 'mnwc' ]
    labels.extend(args.label)

    data = [errors['prst'], errors['mnwc']]

    for l in args.label:
        data.append(errors[l])

    for i,m in enumerate(data):
        for j,lt in enumerate(m):
            data[i][j] = np.mean(data[i][j])

    plot_mae(data, labels, title='MAE over {} predictions'.format(len(predictions['mnwc']['data'])))

    plt.pause(0.001)
    input("Press [enter] to stop")

if __name__ == "__main__":
    args = parse_command_line()

    predictions, errors = predict_many(args)
    plot_results(args, predictions, errors)

