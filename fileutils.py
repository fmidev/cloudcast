import numpy as np
import glob
import sys
import datetime
import cv2
import os
from scipy import ndimage
from PIL import Image, ImageEnhance
from tensorflow import keras
from gributils import *
from osgeo import gdal,osr
from preprocess import *

INPUT_DIR = '/home/partio/cloudnwc/effective_cloudiness/data/'


def get_model_name(args):
    return '{}-{}-{}-{}'.format(args.model, 
                                args.loss_function,
                                args.n_channels,
                                args.preprocess)


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



def create_generators(start_date, stop_date, **kwargs):
    filenames = read_filenames(start_date, stop_date)
    assert(len(filenames) > 0)

    n_channels = int(kwargs.get('n_channels', 1))
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
        self.n_channels = int(kwargs.get('n_channels', 1))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.preprocess = kwargs.get('preprocess', '')
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
            x.append(preprocess_many(read_gribs(d[0:self.n_channels]), self.preprocess))
            y.append(preprocess_many(read_gribs(d[1:self.n_channels+1]), self.preprocess))

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
            x.append(preprocess_single(read_grib(d[0]), self.preprocess))
            y.append(preprocess_single(read_grib(d[1]), self.preprocess))

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

