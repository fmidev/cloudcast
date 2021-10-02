from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
from model import *
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

MODEL="convlstm"
#IMG_SIZE=(928,928)
#IMG_SIZE=(768,768)
#IMG_SIZE=(256,256)
IMG_SIZE=(128,128)
TRAIN_SERIES_LENGTH = 6
CONVLSTM = False

model_file = 'models/{}_{}_{}x{}_{}'.format(MODEL, LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1], TRAIN_SERIES_LENGTH)
print(f"Loading {model_file}")

m = None
if LOSS_FUNCTION == "ssim":
    m = load_model(model_file, compile=False)
else:
    m = load_model(model_file)



def sharpen(data, factor):
    assert(data.shape == (1, 128, 128, 1))
    im = Image.fromarray(np.squeeze(data) * 255)
    im = im.convert('L')

    enhancer = ImageEnhance.Sharpness(im)
    sharp = np.array(enhancer.enhance(factor)) / 255.0
    return np.expand_dims(sharp, [0,3])


def process_img(img):
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=IMG_SIZE)
    return img


def read_img_from_file(filename):
    print(f"reading {filename}")
    img = tf.io.read_file('{}/{}.jpg'.format('/home/partio/tmp/cloudnwc-jpeg/eval', filename))
    return process_img(img)


def read_img_from_memory(mem):
    return process_img(mem)


def infer(img):
    img = tf.expand_dims(img, axis=0)
    prediction = m.predict(img)
    pred = np.squeeze(prediction, axis=0) * 255
    return pred


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
        predicted_frame = sharpen(predicted_frame, 3)

        image_series = np.concatenate((image_series, predicted_frame), axis=0)

    return image_series


def plot_convlstm(ground_truth, predictions):
    fig, axes = plt.subplots(3, ground_truth.shape[0], figsize=(14, 8), constrained_layout=True)

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


def plot_unet(start_date):

    fig, axes = plt.subplots(4, 3, figsize=(14, 10))

    pred = None

    for idx, row in enumerate(axes):

        if idx == 0:
            ground_truth = read_img_from_file(start_date.strftime('%Y%m%dT%H%M'))

            row[0].imshow(ground_truth, cmap='gray')
            row[0].axis("off")
            row[0].set_title(f"Ground truth {idx*15} min")
            row[1].axis("off")
            row[2].axis("off")

            continue

        if pred is None:
            pred = infer(ground_truth)
        else:
            pred = infer(pred/255)

        ground_truth = read_img_from_file(start_date.strftime('%Y%m%dT%H%M'))

        diff = (ground_truth - pred).numpy() / 255

        print(diff.min(), diff.max(), diff.mean())
        row[0].imshow(ground_truth, cmap='gray')
        row[1].imshow(pred, cmap='gray')
        r2 = row[2].imshow(diff, cmap='RdYlBu_r')
        plt.colorbar(r2, ax=row[2])

        row[0].axis("off")
        row[1].axis("off")
        row[2].axis("off")

        row[0].set_title(f"Ground truth {idx*15} min")
        row[1].set_title(f"Predicted {idx*15} min")
        row[2].set_title(f"observed - predicted {idx*15} min")


        start_date = start_date + timedelta(minutes=15)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()


start_date=datetime.strptime('20210101T0045', '%Y%m%dT%H%M')

if CONVLSTM:

    image_series = read_image_series(start_date, 8)

    predictions = predict_from_series(image_series[:4], 4)


    plot_convlstm(image_series[3:], predictions[3:])

    for d in predictions[4:]:
        save_to_file()
else:
   plot_unet(start_date)

