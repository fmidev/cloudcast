import numpy as np
import glob
import sys
import datetime
#import tensorflow as tf
import cv2
import os
from scipy import ndimage
from PIL import Image, ImageEnhance
from tensorflow import keras
from gributils import *

INPUT_DIR = '/home/partio/cloudnwc/effective_cloudiness/data/'


def to_binary_mask(arr):
    arr[arr < 0.1] = 0.0
    arr[arr > 0] = 1.0

    return arr

def preprocess_many(imgs, img_size):
    ds = []

    for img in imgs:
        ds.append(preprocess_single(img, img_size))

    return np.asarray(ds)


def preprocess_single(img, img_size, kernel_size = (3,3), num_classes = 10):

    kernel1 = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    kernel1 = np.expand_dims(kernel1, axis=2)

#    print(f"Preprocessing with {kernel_size} kernel and rounding to {num_classes} classes")

    conv = ndimage.convolve(img, kernel1, mode='constant', cval=0.0)
    #conv = (np.around((100 * conv) / num_classes, decimals=0)*num_classes)/100

    conv = to_binary_mask(conv)


#    im = Image.fromarray(img.squeeze() * 255)
#    im = im.convert('L')
#    im.save("original.jpg")

#    im = Image.fromarray(conv.squeeze() * 255)
#    im = im.convert('L')
#    im.save("convoluted.jpg")

#    print(img)
#    print(conv)
#    print(np.histogram(img))
#    print(np.histogram(conv))
#    sys.exit(1)

    if img_size is not None:
        conv = np.expand_dims(cv2.resize(conv, dsize=img_size, interpolation=cv2.INTER_LINEAR), axis=2)

 
    return conv #.astype(np.float16)


def read_filenames(start_time, stop_time, producer='nwcsaf'):
    print(f'Input directory: {INPUT_DIR}/{producer}')

    files = sorted(glob.glob(f'{INPUT_DIR}/{producer}/**/*.grib2', recursive=True))

    start_date = int(start_time.strftime("%Y%m%d"))
    stop_date = int(stop_time.strftime("%Y%m%d"))
    filenames = []

    for f in files:
        datetime = int(f.split('/')[-1][0:8])
        if datetime >= start_date and datetime < stop_date:
            filenames.append(f)

    return filenames


def sharpen(data, factor):
#    assert(data.shape == (1,) + IMG_SIZE + (1,))
    im = Image.fromarray(np.squeeze(data) * 255)
    im = im.convert('L')

    enhancer = ImageEnhance.Sharpness(im)
    sharp = np.array(enhancer.enhance(factor)) / 255.0
    return np.expand_dims(sharp, [0,3])


def get_filename(time, producer = 'nwcsaf', analysis_time=None):
    if producer == 'nwcsaf':
        return '{}/nwcsaf/{}_nwcsaf_effective-cloudiness.grib2'.format(INPUT_DIR, time.strftime('%Y/%m/%d/%Y%m%dT%H%M%S'))
    if producer == 'mnwc':
        if analysis_time is None:
            # return newest possible
            return '{}/mnwc/{}.grib2'.format(INPUT_DIR, time.strftime('%Y%m%d%H00+000h%Mm'))
        else:
            lt = (time - analysis_time)
            lt_h = int(lt.total_seconds() // 3600)
            lt_m = int(lt.total_seconds() // 60 % 60)
            return '{}/mnwc/{}00+{:03d}h{:02d}m.grib2'.format(INPUT_DIR, analysis_time.strftime('%Y%m%d%H'), lt_h, lt_m)

def read_time(time, producer='nwcsaf', analysis_time=None):
    return read_grib(get_filename(time, producer, analysis_time))

def read_times(times, producer='nwcsaf', analysis_time=None):
    data = []
    for time in times:
        try:
            data.append(read_grib(get_filename(time, producer, analysis_time)))
        except FileNotFoundError as e:
            pass

    return data



def plot_hist(hist, model_dir):
    print(hist.history)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/accuracy.png'.format(model_dir))

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/loss.png'.format(model_dir))


#def create_train_val_split_timeseries(dataset):
#
#    # Split into train and validation sets using indexing to optimize memory.
#    indexes = np.arange(dataset.shape[0])
#    np.random.shuffle(indexes)
#    train_index = indexes[: int(0.9 * dataset.shape[0])]
#    val_index = indexes[int(0.9 * dataset.shape[0]) :]
#    train_dataset = dataset[train_index]
#    val_dataset = dataset[val_index]
#
#    # We'll define a helper function to shift the frames, where
#    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
#    def create_shifted_frames(data):
#        x = data[:, 0 : data.shape[1] - 1, :, :]
#        y = data[:, 1 : data.shape[1], :, :]
#        return x, y
#
#
#    # Apply the processing function to the datasets.
#    x_train, y_train = create_shifted_frames(train_dataset)
#    x_val, y_val = create_shifted_frames(val_dataset)
#
#    # Inspect the dataset.
#    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
#    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
#
#    return (x_train, y_train, x_val, y_val)


def create_train_val_split(dataset, train_history_len=1):

    assert(train_history_len is not None)

    if dataset.shape[0] % 2 == 1:
        dataset = dataset[:-1]

    n_split = dataset.shape[0] / (train_history_len + 1)
    dataset = np.asarray(np.split(dataset, n_split))

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



def time_of_year_and_day(datetime):
    day = 24*60*60
    year = 365.2425 * day

    tod = np.sin(datetime.timestamp() * (2 * np.pi / day))
    toy = np.cos(datetime.timestamp() * (2 * np.pi / year))

    return tod, toy

def create_generators(start_date, stop_date, **kwargs):
    filenames = read_filenames(start_date, stop_date)
    assert(len(filenames) > 0)
    n_channels = kwargs.get('n_channels', 1)
    out = kwargs.get('output_is_timeseries', False)

    datasets = []

    if not out:
        i = 0

        while i < len(filenames) - 1:
            datasets.append([filenames[i], filenames[i+1]])
            i += 2
    else:
        i = 0

        while i < (len(filenames) - (n_channels + 1)):
            ds_files = []
            for j in range(n_channels + 1):
                ds_files.append(filenames[i + j])
            datasets.append(ds_files)
            i += (n_channels + 1)

    np.random.shuffle(datasets)

    test_val_split = (np.floor(len(datasets) * 0.9)).astype(np.int)
    train = EffectiveCloudinessGenerator(datasets[0:test_val_split], **kwargs)
    val = EffectiveCloudinessGenerator(datasets[test_val_split:-1], **kwargs)

    return train, val


class EffectiveCloudinessGenerator(keras.utils.Sequence):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.n_channels = kwargs.get('n_channels', 1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.img_size = kwargs.get('img_size', (256,256))
        self.initial = True
        self.include_time = kwargs.get('include_time', False)
        self.output_is_timeseries = kwargs.get('output_is_timeseries', False)
        assert(self.n_channels > 0)


    def __len__(self):
        return (np.floor(len(self.dataset) / self.batch_size)).astype(np.int)

    def __getitem__(self, i):
        if not self.output_is_timeseries:
            return self.create_single_output_series(i)
        else:
            return self.create_timeseries_output(i)

    def create_timeseries_output(self, i):
        ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for d in ds:
            x.append(preprocess_many(read_gribs(d[0:self.n_channels]), self.img_size))
            y.append(preprocess_many(read_gribs(d[1:self.n_channels+1]), self.img_size))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.include_time:
            for i,f in enumerate(batch_x):
                datetime_str = os.path.filename(f).split('_')[0]

                tod, toy = time_of_year_and_day(datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S'))

                np.append(x[i], np.full(self.img_size, tod, dtype=np.float32), axis=3)
                np.append(x[i], np.full(self.img_size, toy, dtype=np.float32), axis=3)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y

    def create_single_output_series(self, i):
        ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for d in ds:
            x.append(preprocess_single(read_grib(d[0]), self.img_size))
            y.append(preprocess_single(read_grib(d[1]), self.img_size))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.include_time:
            for i,f in enumerate(batch_x):
                datetime_str = os.path.filename(f).split('_')[0]

                tod, toy = time_of_year_and_day(datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S'))

                np.append(x[i], np.full(self.img_size, tod, dtype=np.float32), axis=3)
                np.append(x[i], np.full(self.img_size, toy, dtype=np.float32), axis=3)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y



# datetime ring buffer

class TimeseriesGenerator:
    def __init__(self, start_date, frames_prev, frames_next, step):
        self.date = start_date
        self.frames_prev = frames_prev
        self.frames_next = frames_next
        self.step = step
        self.times = [start_date]
        self.create()
    def __iter__(self):
        while True:
            yield self.times
            self.create()
    def __next__(self):
        return_value = self.times
        self.create()
        return return_value
    def create(self):
        if len(self.times) > 1:
            self.times.pop(0)
        while len(self.times) < 1 + -1 * self.frames_prev + self.frames_next:
            self.times.append(self.times[-1] + self.step)

