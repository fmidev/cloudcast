import requests
import eccodes as ecc
import numpy as np
import pathlib
import glob

from tensorflow.keras.models import save_model # import datasets, models, layers
from PIL import Image
from model import *

IMG_SIZE = (928,928)

def read_input_disk():

    input_dir = pathlib.Path(f'/home/partio/tmp/cloudnwc-jpeg/')
    print(input_dir)

    x_files = glob.glob(f'{input_dir}/train/*.jpg')
    y_files = glob.glob(f'{input_dir}/test/*.jpg')

    files_ds = tf.data.Dataset.from_tensor_slices((x_files, y_files))

    def process_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=IMG_SIZE)
        return img

    files_ds = files_ds.map(lambda x, y: (process_img(x), process_img(y))).batch(1)

    return files_ds


train_ds = read_input_disk()

m = unet(input_size=IMG_SIZE + (1,))

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/{}_{}x{}/cp.ckpt'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1]),
                                                 save_weights_only=True,
                                                 verbose=1)

hist = m.fit(train_ds, epochs=30, batch_size=32, callbacks=[cp_callback])


save_model(m, 'models/{}_{}x{}'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1]))
