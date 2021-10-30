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

CONVLSTM = False

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

    args = parser.parse_args()

    if args.label is not None:
        args.model, args.loss_function, args.n_channels, args.preprocess = args.label.split('-')

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args

if __name__ == "__main__":
    args = parse_command_line()

    model_file = 'models/{}'.format(get_model_name(args))

    if args.model == "convlstm":
        CONVLSTM=True

    print(f"Loading {model_file}")

    m = load_model(model_file, compile=False)



def infer_many(seed, num_predictions):
    predictions = []

    for i in range(num_predictions):
        if len(predictions) > 0:
            predictions.append(infer(predictions[-1]))
        else:
            predictions.append(infer(seed))

    return np.asarray(predictions)



def infer(img):
    img = tf.expand_dims(img, axis=0)
    prediction = m.predict(img)
    pred = np.squeeze(prediction, axis=0)
    return pred


def read_images(times, series, preprocess_label):
    datakeys = series.keys()

    new_series = {}
    for t in times:
        if t in datakeys:
            new_series[t] = series[t]
        else:
            new_series[t] = preprocess_single(read_time(t, producer="nwcsaf"), preprocess_label)

    return new_series


def predict_from_series(dataseries, num):
    for _ in range(num):
        pred = m.predict(np.expand_dims(dataseries, axis=0))
        pred = np.squeeze(pred, axis=0)
        predicted_frame = np.expand_dims(pred[-1, ...], axis=0)

#        predicted_frame = sharpen(predicted_frame, 2)
#        print(np.min(predicted_frame), np.mean(predicted_frame), np.max(predicted_frame))
#        print(np.histogram(predicted_frame))
        dataseries = np.concatenate((dataseries, predicted_frame), axis=0)

    return dataseries[-num:]


def plot_unet(args):

    pred = None
    initial = None

    mae_persistence = []
    mae_prediction = []

    time_gen = TimeseriesGenerator(args.start_date, 0, 5, timedelta(minutes=15))

    times = next(time_gen)
    gt = preprocess_many(read_times(times, producer='nwcsaf'), args.preprocess)
    mnwc = preprocess_many(read_times(times, producer='mnwc'), args.preprocess)
    cloudcast = infer_many(gt[0], len(times))
    initial = np.copy(gt[0])
    plot_timeseries([gt, cloudcast, mnwc], ['ground truth', 'cloudcast', 'mnwc'])

    mae_persistence = []
    mae_cloudcast = []
    mae_mnwc = []

    for i,t in enumerate(gt):
        mae_persistence.append(mean_absolute_error(t.flatten(), initial.flatten()))
        mae_cloudcast.append(mean_absolute_error(t.flatten(), cloudcast[i].flatten()))
        mae_mnwc.append(mean_absolute_error(t.flatten(), mnwc[i].flatten()))

    plot_mae([mae_persistence, mae_cloudcast, mae_mnwc],['persistence', 'cloudcast', 'mnwc'])


if CONVLSTM:

    step = timedelta(minutes=15)
    history_len=5
    prediction_len=6
    gen = TimeseriesGenerator(args.start_date, -history_len, prediction_len, step)

    mae_prediction = []
    mae_persistence = []
    mae_mnwc = []
    image_series = {}

    for i in range(prediction_len+1):
        mae_prediction.append([])
        mae_persistence.append([])
        mae_mnwc.append([])

    initial=True

    for times in gen:
        if times[-1] == args.stop_date:
            break

        print("Reading data between {} and {}...".format(times[0], times[-1]))

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

        analysis_time = times[history_len+1]

        mnwc = preprocess_many(read_times(times[history_len+1:], producer='mnwc', analysis_time=analysis_time.replace(minute=0)), args.preprocess)

        print(future.shape, predictions.shape, mnwc.shape)
        if initial:
            plot_convlstm(future, predictions, mnwc)
            initial=False

        persistence = history[-1]
        for i,(o,p,mn) in enumerate(zip(future, predictions, mnwc)):
            mae_prediction[i].append(mean_absolute_error(o.flatten(), p.flatten()))
            mae_persistence[i].append(mean_absolute_error(o.flatten(), persistence.flatten()))
            mae_mnwc[i].append(mean_absolute_error(o.flatten(), mn.flatten()))


    num_predictions = len(mae_persistence[0])
    for i in range(len(mae_persistence)):
        mae_persistence[i] = np.mean(mae_persistence[i])
        mae_prediction[i] = np.mean(mae_prediction[i])
        mae_mnwc[i] = np.mean(mae_mnwc[i])

    plot_mae([mae_persistence, mae_prediction, mae_mnwc],['persistence', 'cloudcast', 'mnwc'], step)

else:
    plot_unet(args)

