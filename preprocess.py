import numpy as np
import glob
import sys
import eccodes as ecc
import datetime
#import tensorflow as tf
import cv2
from scipy import ndimage
from PIL import Image


INPUT_DIR = '/home/partio/cloudnwc/effective_cloudiness/data/grib2'


def preprocess_many(imgs, img_size):
    ds = []

    for img in imgs:
        ds.append(preprocess_single(img, img_size))

    return np.asarray(ds)


def preprocess_single(img, img_size, kernel_size = (3,3), num_classes = 10):

    kernel1 = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    kernel1 = np.expand_dims(kernel1, axis=2)

#    print(f"Preprocessing with {kernel_size} kernel and rounding to {num_classes} classes")

    ds = []

    conv = ndimage.convolve(img, kernel1, mode='constant', cval=0.0)
    conv = (np.around((100 * conv) / num_classes, decimals=0)*num_classes)/100

#        im = Image.fromarray(img.squeeze() * 255)
#        im = im.convert('L')
#        im.save("original.jpg")

#        im = Image.fromarray(conv.squeeze() * 255)
#        im = im.convert('L')
#        im.save("convoluted.jpg")

#        print(img)
#        print(conv)
#        print(np.histogram(img))
#        print(np.histogram(conv))
#        break

    if img_size is not None:
        conv = np.expand_dims(cv2.resize(conv, dsize=img_size, interpolation=cv2.INTER_LINEAR), axis=2)
#       conv = tf.image.resize(conv, size=img_size)

 
    return conv #.astype(np.float16)


def read_grib(file_path, message_no = 0):
    with open(file_path) as fp:
        gh = ecc.codes_new_from_file(fp, ecc.CODES_PRODUCT_GRIB)
        #year = ecc.codes_get(gh, "year")
        #month = ecc.codes_get(gh, "month")
        #day = ecc.codes_get(gh, "day")
        #hour = ecc.codes_get(gh, "hour")
        #minute = ecc.codes_get(gh, "minute")

        ni = ecc.codes_get_long(gh, "Ni")
        nj = ecc.codes_get_long(gh, "Nj")

        data = np.asarray(ecc.codes_get_double_array(gh, "values"), dtype=np.float32).reshape(nj, ni)
        data = data / 100.0 #* 255 # to mimick an image with one (gray) channel
        if ecc.codes_get(gh, "jScansPositively"):
            data = np.flipud(data) # image data is +x-y
        data = np.expand_dims(data, axis=2)
        return data


def read_input_grib(start_time, stop_time):
    print(f'Input directory: {INPUT_DIR}')

    files = sorted(glob.glob(f'{INPUT_DIR}/**/*.grib2', recursive=True))

    print("Total number of files: {}".format(len(files)))

    def process_grib(file_path):
        img = read_grib(file_path)
#        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    files_ds = []

    start_date = start_time.strftime("%Y%m%d")
    stop_date = stop_time.strftime("%Y%m%d")

    i = 0
    for f in files:
        datetime = f.split('/')[-1][0:14]
        if datetime >= start_date and datetime < stop_date:
            i = i + 1
            if i % 1000 == 0:
                print(f'Read {i}')

            files_ds.append(process_grib(f))

    print(f'Read {i}')
    if len(files_ds) == 0:
        print("No files found")

    return np.asarray(files_ds)


def sharpen(data, factor):
    assert(data.shape == (1,) + IMG_SIZE + (1,))
    im = Image.fromarray(np.squeeze(data) * 255)
    im = im.convert('L')

    enhancer = ImageEnhance.Sharpness(im)
    sharp = np.array(enhancer.enhance(factor)) / 255.0
    return np.expand_dims(sharp, [0,3])


def read_time(time):
    return read_grib('{}/{}_nwcsaf_effective-cloudiness.grib2'.format(INPUT_DIR, time.strftime('%Y/%m/%d/%Y%m%dT%H%M%S')))


def create_train_val_split_timeseries(dataset):

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y


    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return (x_train, y_train, x_val, y_val)


def create_train_val_split(dataset, train_history_len=1):

    assert(train_history_len is not None)

    if dataset.shape[0] % 2 == 1:
        dataset = dataset[:-1]

    n_split = dataset.shape[0] / (train_history_len + 1)
    dataset = np.asarray(np.split(dataset, n_split))

    print(dataset.shape)
    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def split_to_train_test(data):
        x = data[:, 0 : data.shape[1] - 1, :, :].squeeze(1)
        y = data[:, -1, :, :]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = split_to_train_test(train_dataset)
    x_val, y_val = split_to_train_test(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return (x_train, y_train, x_val, y_val)


def create_dataset(start_time, stop_time, img_size=None, preprocess=False):
    print(f"Creating dataset with time range {start_time} to {stop_time}")
    ds = read_input_grib(start_time, stop_time)

    print(f"Dataset shape: {ds.shape}")

    if preprocess:
        ds = preprocess_many(ds, img_size)
        print(f'Dataset shape: {ds.shape}')

    return ds


