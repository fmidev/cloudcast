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
DEM = None
LSM = None

def get_model_name(args):
    return '{}-{}-{}-{}-{}-{}'.format(args.model,
                                args.loss_function,
                                args.n_channels,
                                args.include_datetime,
                                args.include_environment_data,
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


def create_datetime(datetime, img_size):
    tod, toy = time_of_year_and_day(datetime)
    tod = np.expand_dims(np.full(img_size, tod, dtype=np.float32), axis=-1)
    toy = np.expand_dims(np.full(img_size, toy, dtype=np.float32), axis=-1)

    return tod, toy


def process_lsm(LSM):

    # Value	Label
    # 11	Post-flooding or irrigated croplands (or aquatic)
    # 14	Rainfed croplands
    # 20	Mosaic cropland (50-70%) / vegetation (grassland/shrubland/forest) (20-50%)
    # 30	Mosaic vegetation (grassland/shrubland/forest) (50-70%) / cropland (20-50%)
    # 40	Closed to open (>15%) broadleaved evergreen or semi-deciduous forest (>5m)
    # 50	Closed (>40%) broadleaved deciduous forest (>5m)
    # 60	Open (15-40%) broadleaved deciduous forest/woodland (>5m)
    # 70	Closed (>40%) needleleaved evergreen forest (>5m)
    # 90	Open (15-40%) needleleaved deciduous or evergreen forest (>5m)
    # 100	Closed to open (>15%) mixed broadleaved and needleleaved forest (>5m)
    # 110	Mosaic forest or shrubland (50-70%) / grassland (20-50%)
    # 120	Mosaic grassland (50-70%) / forest or shrubland (20-50%)
    # 130	Closed to open (>15%) (broadleaved or needleleaved, evergreen or deciduous) shrubland (<5m)
    # 140	Closed to open (>15%) herbaceous vegetation (grassland, savannas or lichens/mosses)
    # 150	Sparse (<15%) vegetation
    # 160	Closed to open (>15%) broadleaved forest regularly flooded (semi-permanently or temporarily) - Fresh or brackish water
    # 170	Closed (>40%) broadleaved forest or shrubland permanently flooded - Saline or brackish water
    # 180	Closed to open (>15%) grassland or woody vegetation on regularly flooded or waterlogged soil - Fresh, brackish or saline water
    # 190	Artificial surfaces and associated areas (Urban areas >50%)
    # 200	Bare areas
    # 210	Water bodies
    # 220	Permanent snow and ice
    # 230	No data (burnt areas, clouds,)

    # forest
    LSM[np.logical_and(LSM >= 40, LSM >= 100)] = 0

    # urban
    LSM[LSM == 190] = 1

    # sparse / bare
    LSM[np.logical_or(LSM == 150, LSM == 200)] = 2

    # permanent snow
    LSM[LSM == 220] = 3

    # water
    LSM[LSM == 210] = 4

    # agriculture
    LSM[np.logical_or(LSM <= 14, LSM == 20)] = 5

    # rest
    LSM[LSM > 5] = 6

    return LSM



def create_environment_data(preprocess_label):
    global LSM, DEM

    if LSM is not None and DEM is not None:
        return LSM, DEM

    tokens = preprocess_label.split(',')

    proc=['standardize=true']

    for t in tokens:
        k,v = t.split('=')

        if k in ('img_size',):
            proc.append(t)

    proc = ','.join(proc)
    lsm_file = '{}/static/LSM-cloudcast.tif'.format(INPUT_DIR)
    dem_file = '{}/static/DEM-cloudcast.tif'.format(INPUT_DIR)

    print (f"Reading {lsm_file}")
    raster = gdal.Open(lsm_file)
    LSM = raster.GetRasterBand(1).ReadAsArray()
    LSM = process_lsm(LSM)

    LSM = preprocess_single(LSM, proc)

    print (f"Reading {dem_file}")
    raster = gdal.Open(dem_file)
    DEM = raster.GetRasterBand(1).ReadAsArray()
    DEM = preprocess_single(DEM, proc)

    raster = None

    return LSM, DEM

class EffectiveCloudinessGenerator(keras.utils.Sequence):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.n_channels = int(kwargs.get('n_channels', 1))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.preprocess = kwargs.get('preprocess', '')
        self.initial = True
        self.include_datetime = kwargs.get('include_datetime', False)
        self.include_environment_data = kwargs.get('include_environment_data', False)
        self.output_is_timeseries = kwargs.get('output_is_timeseries', False)

        assert(self.n_channels > 0)


    def __len__(self):
        return (np.floor(len(self.dataset) / self.batch_size)).astype(np.int)

    def __getitem__(self, i):
        if not self.output_is_timeseries:
            return self.create_single_output_series(i)
        else:
            return self.create_timeseries_output(i)

#    def create_datetime(self, filename, img_size):
#        datetime_str = os.path.basename(filename).split('_')[0]
#        return create_datetime(datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S'), img_size)

    def create_timeseries_output(self, i):
        ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for d in ds:
            x.append(preprocess_many(read_gribs(d[0:self.n_channels]), self.preprocess))
            y.append(preprocess_many(read_gribs(d[1:self.n_channels+1]), self.preprocess))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y

    def create_single_output_series(self, i):
        batch_ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for ds in batch_ds:
            x.append([preprocess_single(read_grib(ds[0]), self.preprocess)])
            y.append(preprocess_single(read_grib(ds[1]), self.preprocess))
            dt = datetime.datetime.strptime(os.path.basename(ds[0]).split('_')[0], '%Y%m%dT%H%M%S')

            if self.include_datetime:
                x[-1].extend(create_datetime(dt, get_img_size(self.preprocess)))

            if self.include_environment_data:
                x[-1].extend(create_environment_data(self.preprocess))
            x[-1] = np.stack(x[-1], axis=-1)

        x = np.squeeze(np.asarray(x), axis=-2)
#        x = np.expand_dims(x, axis=1)
#        x = np.repeat(x, 4, axis=1)
        y = np.asarray(y)

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

class TimeseriesGenerator2:
    def __init__(self, start_date, ts_length, step=datetime.timedelta(minutes=15), stop_date=None):
        self.date = start_date
        self.stop_date = stop_date
        self.ts_length = ts_length
        self.step = step
        self.times = [start_date]
        self.create()
    def __iter__(self):
        while self.times[-1] != self.stop_date:
            yield self.times
            self.create()
    def create(self):
        if len(self.times) > 1:
            self.times.pop(0)
        while len(self.times) < self.ts_length: # and (self.stop_date is None or self.times[-1] != self.stop_date):
            self.times.append(self.times[-1] + self.step)


class DataSeries:
    def __init__(self, producer, preprocess = None, single_analysis_time = True):
        self.data_series = {}
        self.producer = producer
        self.preprocess = preprocess
        self.analysis_time = None
        self.single_analysis_time = single_analysis_time

    def read_data():
        return np.asarray(list(self.data_series.values()))

    def read_data(self, times, analysis_time=None):
        datakeys = self.data_series.keys()

        if analysis_time != self.analysis_time and self.single_analysis_time:
            datakeys = []

        new_series = {}
        for t in times:
            if t in datakeys:
                new_series[t] = self.data_series[t]
            else:
                new_series[t] = preprocess_single(read_time(t, self.producer, analysis_time), self.preprocess)

        self.data_series = new_series
        self.analysis_time = analysis_time

        return np.asarray(list(self.data_series.values()))
