from tensorflow.keras.models import load_model
from PIL import Image
from model import *
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#IMG_SIZE=(768,768)

IMG_SIZE=(256,256)
#IMG_SIZE=(128,128)


model_file = 'models/{}_{}x{}'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1])
#model_file = 'models/cloudcast.model'
print(f"Loading {model_file}")

m = load_model(model_file)

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


fig, axes = plt.subplots(4, 3, figsize=(14, 10))

start_date=datetime.strptime('20210101T0045', '%Y%m%dT%H%M')
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
