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
from preprocess import *

CONVLSTM = False

def parse_command_line():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--img_size", action='store', type=str, default='256x256')
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

    model_file = 'models/{}'.format(model_name(args))

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

    return predictions



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
#    union = list(set().union(times, keys))
#    union.sort()

#    new_series = {}
#    for u in union:
#        if u in keys:
#            new_series[u] = series[u]
#        else:
#            new_series[u] = read_grib(u)
#    for time in times:
#        series.append(read_img_from_file(time))

#    return new_series


#def read_image_series(start_date, num):
#    series = []
#    for _ in range(num):
#        series.append(read_grib(get_filename(start_date.strftime('%Y%m%dT%H%M')), producer="nwcsaf"))
#        start_date = start_date + timedelta(minutes=15)
#
#    return np.asarray(series)


def predict_from_series(dataseries, num):
    ret = []
    for _ in range(num):
        pred = m.predict(np.expand_dims(dataseries, axis=0))
        pred = np.squeeze(pred, axis=0)
        predicted_frame = to_binary_mask(np.expand_dims(pred[-1, ...], axis=0))

#        predicted_frame = sharpen(predicted_frame, 2)
        print(np.min(predicted_frame), np.mean(predicted_frame), np.max(predicted_frame))
        print(np.histogram(predicted_frame))
        dataseries = np.concatenate((dataseries, predicted_frame), axis=0)

    return dataseries[-num:]


def plot_convlstm(ground_truth, predictions, mnwc):
    fig, axes = plt.subplots(4, ground_truth.shape[0], figsize=(16, 8), constrained_layout=True)

    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(ground_truth[idx]), cmap='gray')
        ax.set_title(f'ground truth frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(predictions[idx]), cmap='gray')
        ax.set_title(f'prediction frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[2]):
        ax.imshow(np.squeeze(mnwc[idx]), cmap='gray')
        ax.set_title(f'mnwc frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[3]):
        r = ax.imshow(np.squeeze(ground_truth[idx] - predictions[idx]), cmap='bwr')

        if idx == 0:
            plt.colorbar(r, ax=axes[3])
        ax.set_title(f'diff frame {idx}')
        ax.axis('off')

    plt.show()


def plot_mae(data, labels, step=timedelta(minutes=15)):
    print(data)
    print(labels)
    assert(len(data) == len(labels))
    fig = plt.figure()
    ax = plt.axes()

    x = list(map(lambda x: step * x, range(len(data[0]))))
    x = list(map(lambda x: '{}m'.format(int(x.total_seconds() / 60)), x))

    for i,mae in enumerate(data):
        ax.plot(x, mae, label=labels[i])

    plt.legend()
    plt.title(f'mae over {len(data[0])} predictions')
    plt.show()


def plot_unet(args):

    fig, axes = plt.subplots(5, 3, figsize=(14, 8))

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

    mae_persistence = []
    mae_cloudcast = []
    mae_mnwc = []

    for i,t in enumerate(gt):
        mae_persistence.append(mean_absolute_error(t.flatten(), initial.flatten()))
        mae_cloudcast.append(mean_absolute_error(t.flatten(), cloudcast[i].flatten()))
        mae_mnwc.append(mean_absolute_error(t.flatten(), mnwc[i].flatten()))

    plot_mae([mae_persistence, mae_cloudcast, mae_mnwc],['persistence', 'cloudcast', 'mnwc'])

    for i, row in enumerate(axes):
        time = times[i]

        if i == 0:
            row[0].imshow(gt[0], cmap='gray')
            row[0].axis("off")
            row[0].set_title(f"Ground truth 0 min")
            row[1].axis("off")
            row[2].axis("off")

            continue

        diff = (gt[i] - cloudcast[i])

        row[0].imshow(gt[i]*255, cmap='gray')
        row[1].imshow(cloudcast[i]*255, cmap='gray')
        r2 = row[2].imshow(diff, cmap='RdYlBu_r')
        plt.colorbar(r2, ax=row[2])

        row[0].axis("off")
        row[1].axis("off")
        row[2].axis("off")

        row[0].set_title(f"Ground truth {i*15} min")
        row[1].set_title(f"Predicted {i*15} min")
        row[2].set_title(f"observed - predicted {i*15} min")


    plt.show()



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
#        assert(mnwc.shape[0] > 0)

        print(future.shape, predictions.shape, mnwc.shape)
        plot_convlstm(future, predictions, mnwc)

        persistence = history[-1]
        for i,(o,p,mn) in enumerate(zip(future, predictions, mnwc)):
            mae_prediction[i].append(mean_absolute_error(o.flatten(), p.flatten()))
            mae_persistence[i].append(mean_absolute_error(o.flatten(), persistence.flatten()))
            mae_mnwc[i].append(mean_absolute_error(o.flatten(), mn.flatten()))

        break
    num_predictions = len(mae_persistence[0])
    for i in range(len(mae_persistence)):
        mae_persistence[i] = np.mean(mae_persistence[i])
        mae_prediction[i] = np.mean(mae_prediction[i])
        mae_mnwc[i] = np.mean(mae_mnwc[i])

    plot_mae([mae_persistence, mae_prediction, mae_mnwc],['persistence', 'cloudcast', 'mnwc'], step)

else:
    plot_unet(args)

