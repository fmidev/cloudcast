from tensorflow.keras.models import load_model
from model import *
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
    parser.add_argument("--loss_function", action='store', type=str, default='MeanSquaredError')
    parser.add_argument("--model", action='store', type=str, default='unet')
    parser.add_argument("--n_channels", action='store', type=int, default=1)
    parser.add_argument("--preprocess", action='store', type=str, default='conv=3,classes=10')
    parser.add_argument("--label", action='store', type=str)
    parser.add_argument("--area", action='store', type=str, default='Scandinavia')
    parser.add_argument("--include_datetime", action='store_true', default=False)
    parser.add_argument("--include_environment_data", action='store_true', default=False)

    args = parser.parse_args()

    if args.label is not None:
        args.model, args.loss_function, args.n_channels, args.include_datetime, args.include_environment_data, args.preprocess = args.label.split('-')
        args.n_channels = int(args.n_channels)

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args

if __name__ == "__main__":
    args = parse_command_line()

    model_file = 'models/{}'.format(get_model_name(args))

    print(f"Loading {model_file}")

    m = load_model(model_file, compile=False)



def infer_many(orig, num_predictions, datetime_weights=None, environment_weights=None):
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

        pred = infer(data)
        predictions.append(pred)

    return np.asarray(predictions)



def infer(img):
    img = np.expand_dims(img, axis=0)
    prediction = m.predict(img)
    pred = np.squeeze(prediction, axis=0)
    return pred



def predict_from_series(dataseries, num):
#    for _ in range(num):
        pred = m.predict(np.expand_dims(dataseries, axis=0))
        pred = np.squeeze(pred, axis=0)
        return pred
#        sys.exit(1)
#        pred = np.squeeze(pred, axis=0)
#        predicted_frame = np.expand_dims(pred[-1, ...], axis=0)

#        predicted_frame = sharpen(predicted_frame, 2)
#        print(np.min(predicted_frame), np.mean(predicted_frame), np.max(predicted_frame))
#        print(np.histogram(predicted_frame))
#        dataseries = np.concatenate((dataseries, predicted_frame), axis=0)

#    return dataseries[-num:]


def predict_unet(args):
    initial = None

    pred_gt = []
    pred_cc = []
    pred_mnwc = []

    mae_prst = []
    mae_cc = []
    mae_mnwc = []

    for i in range(PRED_LEN):
        mae_prst.append([])
        mae_cc.append([])
        mae_mnwc.append([])

    time_gen = TimeseriesGenerator(args.start_date, PRED_LEN + args.n_channels, step=PRED_STEP, stop_date=args.stop_date)

    environment_weights = None

    gt_ds = DataSeries("nwcsaf", args.preprocess)
    mnwc_ds = DataSeries("mnwc", args.preprocess)

    for times in time_gen:
        history = times[:args.n_channels]
        leadtimes = times[args.n_channels:]

        print("Using history {} to predict {}".format(
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), history)),
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), leadtimes))
        ))

        gt = gt_ds.read_data(times)

        mnwc = mnwc_ds.read_data(leadtimes, leadtimes[0].replace(minute=0))
        initial = np.copy(gt[args.n_channels - 1])

        if np.isnan(gt).any():
            print("Seed contains missing values, skipping")
            continue

        gt = gt[args.n_channels:]

        datetime_weights = None

        if args.include_datetime:
            datetime_weights = list(map(lambda x: create_datetime(x, get_img_size(args.preprocess)), leadtimes))
        if args.include_environment_data and environment_weights is None:
            environment_weights = create_environment_data(args.preprocess)

        cc = infer_many(gt[:args.n_channels], PRED_LEN, datetime_weights, environment_weights)

        pred_gt.append(gt)
        pred_cc.append(cc)
        pred_mnwc.append(mnwc)

        for i,t in enumerate(gt):
            if np.isnan(t).any():
                continue

            if not np.isnan(mnwc).any():
                mae_mnwc[i].append(mean_absolute_error(t.flatten(), mnwc[i].flatten()))

            mae_prst[i].append(mean_absolute_error(t.flatten(), initial.flatten()))
            mae_cc[i].append(mean_absolute_error(t.flatten(), cc[i].flatten()))


    return [pred_gt, pred_cc, pred_mnwc], [mae_prst, mae_cc, mae_mnwc]

def plot_results(args, predictions, errors):

    pred_gt = predictions[0]
    pred_cc = predictions[1]
    pred_mnwc = predictions[2]

    mae_prst = errors[0]
    mae_cc = errors[1]
    mae_mnwc = errors[2]

    idx = np.random.randint(len(pred_gt))
    plot_timeseries([pred_gt[idx], pred_cc[idx], pred_mnwc[idx]], ['ground truth', 'cloudcast', 'mnwc'], title='Prediction for t0={}'.format(idx * PRED_STEP + args.start_date))

    for i,lt in enumerate(mae_prst):
        mae_prst[i] = np.mean(mae_prst[i])
        mae_cc[i] = np.mean(mae_cc[i])
        mae_mnwc[i] = np.mean(mae_mnwc[i])

    plot_mae([mae_prst, mae_cc, mae_mnwc],['persistence', 'cloudcast', 'mnwc'], title='MAE over {} predictions'.format(len(pred_gt)))


def predict_convlstm(args):

    time_gen = TimeseriesGenerator(args.start_date, PRED_LEN + args.n_channels, step=PRED_STEP, stop_date=args.stop_date)

    pred_gt = []
    pred_cc = []
    pred_mnwc = []

    mae_cc = []
    mae_prst = []
    mae_mnwc = []
#    image_series = {}

    environment_weights = None

    gt_ds = DataSeries("nwcsaf", args.preprocess)
    mnwc_ds = DataSeries("mnwc", args.preprocess)

    for i in range(PRED_LEN):
        mae_cc.append([])
        mae_prst.append([])
        mae_mnwc.append([])

    for times in time_gen:
        history = times[:args.n_channels]
        leadtimes = times[args.n_channels:]

        print("Using history {} to predict {}".format(
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), history)),
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), leadtimes))
        ))

        gt = gt_ds.read_data(times)

        mnwc = mnwc_ds.read_data(leadtimes, leadtimes[0].replace(minute=0))
        initial = np.copy(gt[args.n_channels - 1])

        if np.isnan(gt).any():
            print("Seed contains missing values, skipping")
            continue

        gt = gt[args.n_channels:]

        datetime_weights = None

        if args.include_datetime:
            datetime_weights = list(map(lambda x: create_datetime(x, get_img_size(args.preprocess)), leadtimes))
        if args.include_environment_data and environment_weights is None:
            environment_weights = create_environment_data(args.preprocess)

        #cc = infer_many(gt[:args.n_channels], PRED_LEN, datetime_weights, environment_weights)
        cc = predict_from_series(gt[:args.n_channels], PRED_LEN)

        pred_gt.append(gt)
        pred_cc.append(cc)
        pred_mnwc.append(mnwc)

        for i,t in enumerate(gt):
            if np.isnan(t).any():
                continue

            if not np.isnan(mnwc).any():
                mae_mnwc[i].append(mean_absolute_error(t.flatten(), mnwc[i].flatten()))

            mae_prst[i].append(mean_absolute_error(t.flatten(), initial.flatten()))
            mae_cc[i].append(mean_absolute_error(t.flatten(), cc[i].flatten()))
            print(np.mean(cc[i]))

        continue
        # create a timeseries that consists of history, present, and future
        # first two are used to create a prediction, and the latter one is
        # used to verify the prediction

        dataseries = read_images(times, image_series, args.preprocess)
        assert(len(dataseries) == (1 + history_len + prediction_len))

        # the actual data is in values
        datas = list(dataseries.values())
        history = datas[:history_len+1]
        future = np.asarray(datas[history_len+1:])

        # contains the predicted frames *only*
        predictions = predict_from_series(history, prediction_len)
        assert(predictions.shape[0] == prediction_len)
        assert(predictions.shape[0] == len(future))


    return [pred_gt, pred_cc, pred_mnwc], [mae_prst, mae_cc, mae_mnwc]


if args.model == "convlstm":
    predictions, errors = predict_convlstm(args)
    plot_results(args, predictions, errors)

else:
    predictions, errors = predict_unet(args)
    plot_results(args, predictions, errors)

