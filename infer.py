from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
from model import *
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from preprocess import *

#MODEL="convlstm"
MODEL="unet"
#IMG_SIZE=(928,928)
#IMG_SIZE=(768,768)
IMG_SIZE=(256,256)
IMG_SIZE=(128,128)
TRAIN_SERIES_LENGTH = 8
CONVLSTM = False
#LOSS = "MeanAbsoluteError"
LOSS = "MeanSquaredError"

model_file = 'models/{}_{}_{}x{}'.format(MODEL, LOSS, IMG_SIZE[0], IMG_SIZE[1])

if MODEL == "convlstm":
    CONVLSTM=True
    model_file = 'models/{}_{}_{}x{}_{}'.format(MODEL, LOSS, IMG_SIZE[0], IMG_SIZE[1], TRAIN_SERIES_LENGTH)

print(f"Loading {model_file}")

m = None
if LOSS == "ssim":
    m = load_model(model_file, compile=False)
else:
    m = load_model(model_file)



def sharpen(data, factor):
    assert(data.shape == (1,) + IMG_SIZE + (1,))
    im = Image.fromarray(np.squeeze(data) * 255)
    im = im.convert('L')

    enhancer = ImageEnhance.Sharpness(im)
    sharp = np.array(enhancer.enhance(factor)) / 255.0
    return np.expand_dims(sharp, [0,3])



def read_img_from_memory(mem):
    return process_img(mem)


def infer(img):
    img = tf.expand_dims(img, axis=0)
    prediction = m.predict(img)
    pred = np.squeeze(prediction, axis=0)
    return pred


def read_images(times, series):
    datakeys = series.keys()

    new_series = {}
    for t in times:
        if t in datakeys:
            new_series[t] = series[t]
        else:
            new_series[t] = read_img_from_file(t)

    return new_series
    union = list(set().union(times, keys))
    union.sort()

    new_series = {}
    for u in union:
        if u in keys:
            new_series[u] = series[u]
        else:
            new_series[u] = read_img_from_file(u)
#    for time in times:
#        series.append(read_img_from_file(time))
    return new_series


def read_image_series(start_date, num):
    series = []
    for _ in range(num):
        series.append(read_img_from_file(start_date.strftime('%Y%m%dT%H%M')))
        start_date = start_date + timedelta(minutes=15)

    return np.asarray(series)


def predict_from_series(image_series, num):
    for _ in range(num):
        pred = m.predict(np.expand_dims(image_series, axis=0))
        pred = np.squeeze(pred, axis=0)
        predicted_frame = np.expand_dims(pred[-1, ...], axis=0)
        predicted_frame = sharpen(predicted_frame, 2)

        image_series = np.concatenate((image_series, predicted_frame), axis=0)

    return image_series


def plot_convlstm(ground_truth, predictions):
    fig, axes = plt.subplots(3, ground_truth.shape[0], figsize=(16, 8), constrained_layout=True)

    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(ground_truth[idx]), cmap='gray')
        ax.set_title(f'ground truth frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(predictions[idx]), cmap='gray')
        ax.set_title(f'prediction frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[2]):
        r = ax.imshow(np.squeeze(ground_truth[idx] - predictions[idx]), cmap='bwr')

        if idx == 0:
            plt.colorbar(r, ax=axes[2])
        ax.set_title(f'diff frame {idx}')
        ax.axis('off')

    plt.show()


def save_to_file(data, outfile, datetime):
    assert(outfile[-5:] == 'grib2')

    outfile = f'predicted/{outfile}'
    try:
        os.makedirs(os.path.dirname(outfile))
    except FileExistsError as e:
        pass

    with open(outfile) as fp:
        h = ecc.codes_grib_new_from_samples("redular_ll_sfc_grib2")
        ecc.codes_set(h, "grid_type", "lambert")
        ecc.codes_set(h, 'shapeOfTheEarth', 5)
        ecc.codes_set(h, 'Nx', 928)
        ecc.codes_set(h, 'Ny', 1024)
        ecc.codes_set(h, 'DxInMeters', 3000.4)
        ecc.codes_set(h, 'DyInMeters', 3000.4)
        ecc.codes_set(h, 'jScansPositive', 0)
        ecc.codes_set(h, "latitudeOfFirstPointInDegrees", 47.86)
        ecc.codes_set(h, "longitudeOfFirstPointDegrees", 358.542)
        ecc.codes_set(h, "latin1InDegrees", 63.3)
        ecc.codes_set(h, "latin2InDegrees", 63.3)
        ecc.codes_set(h, "LoVInDegrees", 15)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(datetime.strftime('%Y%m%d')))
        ecc.codes_set(h, "dataTime", int(int(datetime.strftime('%H%M')/100)))
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", 255)
        ecc.codes_set(h, "discipline", 192)
        ecc.codes_set(h, "parameterCategory", 128)
        ecc.codes_set(h, "parameterNumber", 164)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        #ecc.codes_set(h, "missingValue", MISSING)
        #ecc.codes_set(h, "bitmapPresent", 1)

        with open(outfile, 'wb') as fpout:
            ecc.codes_write(h, fpout)
            print(f'Wrote file {outfile}')

        ecc.codes_release(h)

def plot_mae(persistence, prediction):

#    for i in range(len(mae_persistence)):
#        mae_persistence[i] = np.mean(persistence[i])
#        mae_prediction[i] = np.mean(prediction[i])
    assert(len(persistence) == len(prediction))
    print(prediction)
    fig = plt.figure()
    ax = plt.axes()

    x = range(len(prediction))
    ax.plot(x, prediction, label='model')
    ax.plot(x, persistence, label='persistence')
    plt.legend()
    plt.title(f'mae over {len(prediction)} predictions')
    plt.show()


def plot_unet(start_date):

    fig, axes = plt.subplots(5, 3, figsize=(14, 8))

    pred = None
    initial = None

    mae_persistence = []
    mae_prediction = []

    for idx, row in enumerate(axes):

        if idx == 0:
            ground_truth = preprocess_single(read_time(start_date), img_size=IMG_SIZE)
            initial = np.copy(ground_truth)

            row[0].imshow(ground_truth, cmap='gray')
            row[0].axis("off")
            row[0].set_title(f"Ground truth 0 min")
            row[1].axis("off")
            row[2].axis("off")
            start_date = start_date + timedelta(minutes=15)

            continue

        if pred is None:
            pred = infer(ground_truth)
        else:
            pred = infer(pred)

        ground_truth = preprocess_single(read_time(start_date), img_size=IMG_SIZE)
        print(np.mean(ground_truth), np.mean(pred))

        mae_persistence.append(mean_absolute_error(ground_truth.flatten(), initial.flatten()))
        mae_prediction.append(mean_absolute_error(ground_truth.flatten(), pred.flatten()))

        print("mae persistence: {}".format(mae_persistence[:-1]))
        print("mae prediction:  {}".format(mae_prediction[:-1]))

        print(np.histogram(pred))
        print(np.histogram(ground_truth))
        diff = (ground_truth - pred) #/ 255

        row[0].imshow(ground_truth*255, cmap='gray')
        row[1].imshow(pred*255, cmap='gray')
        r2 = row[2].imshow(diff, cmap='RdYlBu_r')
        plt.colorbar(r2, ax=row[2])

        row[0].axis("off")
        row[1].axis("off")
        row[2].axis("off")

        row[0].set_title(f"Ground truth {idx*15} min")
        row[1].set_title(f"Predicted {idx*15} min")
        row[2].set_title(f"observed - predicted {idx*15} min")


        start_date = start_date + timedelta(minutes=15)


    plt.show()

    plot_mae(mae_persistence, mae_prediction)



start_date=datetime.datetime.strptime('20200101T0045', '%Y%m%dT%H%M')

if CONVLSTM:

    history_len = 5
    prediction_len = 12
    gen = TimeseriesGenerator(start_date, -history_len, prediction_len, timedelta(minutes=15))

    mae_prediction = []
    mae_persistence = []
    image_series = {}

    for i in range(prediction_len+1):
        mae_prediction.append([])
        mae_persistence.append([])

    break_date = '20200101T0800' # '20200201T0000'
    for t in gen:
        times = list(map(lambda x: datetime.strftime(x, '%Y%m%dT%H%M'), t))
        if times[-1] == break_date:
            break

        image_series = read_images(times, image_series)
        assert(len(image_series) == (1 + history_len + prediction_len))

        images = list(image_series.values()) #[:history_len+1]
        predictions = predict_from_series(images[:history_len+1], prediction_len)
        assert(len(image_series) == len(predictions))
        #plot_convlstm(image_series[history_len:], predictions[history_len:])
        for i in range(history_len, len(predictions)):
            #mae_prediction[i - history_len].append(mean_absolute_error(image_series[i].flatten(), predictions[i].flatten()))
            mae_prediction[i - history_len].append(mean_absolute_error(images[i].numpy().flatten(), predictions[i].flatten()))
            #mae_persistence[i - history_len].append(mean_absolute_error(image_series[i].flatten(), image_series[history_len].flatten()))
            mae_persistence[i - history_len].append(mean_absolute_error(images[i].numpy().flatten(), images[history_len].numpy().flatten()))

    num_predictions = len(mae_persistence[0])
    for i in range(len(mae_persistence)):
        mae_persistence[i] = np.mean(mae_persistence[i])
        mae_prediction[i] = np.mean(mae_prediction[i])

    fig = plt.figure()
    ax = plt.axes()

    x = range(len(mae_prediction))
    ax.plot(x, mae_prediction, label='convnet')
    ax.plot(x, mae_persistence, label='persistence')
    plt.legend()
    plt.title(f'mae over {num_predictions} predictions')
    plt.show()
#    for d in predictions[4:]:
#        save_to_file()
else:
    plot_unet(start_date)

