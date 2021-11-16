import numpy as np
import glob
import sys
import datetime
import os
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from gributils import *
from osgeo import gdal,osr
from preprocess import *

INPUT_DIR = 'https://lake.fmi.fi/cc_archive'

if os.environ['HOSTNAME'] == 'pansuola':
    INPUT_DIR = '/home/partio/cloudnwc/effective_cloudiness/data/'

DEM = {}
LSM = {}

def get_model_name(args):
    return '{}-{}-{}-{}-{}-{}-{}'.format(args.model,
                                args.loss_function,
                                args.n_channels,
                                args.include_datetime,
                                args.include_environment_data,
                                args.leadtime_conditioning,
                                args.preprocess)


def read_filenames_from_s3(start_time, stop_time, producer):
    print("Getting object listing from s3")
    s3 = boto3.client('s3', endpoint_url='https://lake.fmi.fi', use_ssl=True, config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='cc_archive', Prefix=producer + '/')

    start_date = int(start_time.strftime("%Y%m%d"))
    stop_date = int(stop_time.strftime("%Y%m%d"))
    filenames = []

    for page in pages:
        for item in page['Contents']:
            f = item['Key']

            datetime = int(f.split('/')[-1][0:8])
            if datetime >= start_date and datetime < stop_date:
                filenames.append('https://lake.fmi.fi/cc_archive/{}'.format(f))

    print("Filter matched {} files".format(len(filenames)))
    return filenames


def read_filenames(start_time, stop_time, producer='nwcsaf'):
    print(f'Input directory: {INPUT_DIR}/{producer}')

    if INPUT_DIR[0:4] == 'http':
        return read_filenames_from_s3(start_time, stop_time, producer)

    files = sorted(glob.glob(f'{INPUT_DIR}/{producer}/**/*.grib2', recursive=True))

    start_date = int(start_time.strftime("%Y%m%d"))
    stop_date = int(stop_time.strftime("%Y%m%d"))
    filenames = []

    for f in files:
        datetime = int(f.split('/')[-1][0:8])
        if datetime >= start_date and datetime < stop_date:
            filenames.append(f)

    print("Filter matched {} files".format(len(filenames)))
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
    if producer == 'meps':
        assert(analysis_time is not None)
        ahour = int(analysis_time.strftime('%H'))
        ahour = ahour - ahour % 3
        lt = (time - analysis_time)
        lt_h = int(lt.total_seconds() // 3600)
        lt_m = int(lt.total_seconds() // 60 % 60)
        return '{}/meps/{}{}00+{:03d}h{:02d}m.grib2'.format(INPUT_DIR, analysis_time.strftime('%Y%m%d'), ahour, lt_h, lt_m)



def read_time(time, producer='nwcsaf', analysis_time=None, **kwargs):
    return read_grib(get_filename(time, producer, analysis_time), **kwargs)



def read_times(times, producer='nwcsaf', analysis_time=None):
    data = []
    for time in times:
        try:
            data.append(read_grib(get_filename(time, producer, analysis_time)))
        except FileNotFoundError as e:
            pass

    return data



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



def gdal_read_from_http(url):
    mmap_name = '/vsimem/xxx'

    def read_from_http(url):
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            print(f"Not found: {url}")
            sys.exit(1)
        return r.content

    gdal.FileFromMemBuffer(mmap_name, read_from_http(url))
    return gdal.Open(mmap_name)

def create_onehot_leadtime_conditioning(img_size, depth, active_layer):
    b = np.ones((1,) + img_size)
    return np.expand_dims(np.expand_dims(np.eye(depth)[active_layer], -1), 1) * b


def create_squeezed_leadtime_conditioning(img_size, depth, active_leadtime):
    return np.expand_dims(np.full(img_size, active_leadtime / depth), axis=(0,3))


def create_environment_data(preprocess_label):
    global LSM, DEM

    isize = get_img_size(preprocess_label)
    img_size = '{}x{}'.format(isize[0], isize[1])

    try:
        return LSM[img_size], DEM[img_size]
    except KeyError as e:
        pass

    tokens = preprocess_label.split(',')

    proc='standardize=true,img_size={}'.format(img_size)
    lsm_file = '{}/static/LSM-cloudcast.tif'.format(INPUT_DIR)
    dem_file = '{}/static/DEM-cloudcast.tif'.format(INPUT_DIR)

    print (f"Reading {lsm_file}")

    if lsm_file[0:4] == 'http':
        raster = gdal_read_from_http(lsm_file)
    else:
        raster = gdal.Open(lsm_file)

    LSM[img_size] = raster.GetRasterBand(1).ReadAsArray()
    LSM[img_size] = process_lsm(LSM[img_size])

    LSM[img_size] = preprocess_single(LSM[img_size], proc)

    print (f"Reading {dem_file}")

    raster = None

    if dem_file[0:4] == 'http':
        raster = gdal_read_from_http(dem_file)
    else:
        raster = gdal.Open(dem_file)

    DEM[img_size] = raster.GetRasterBand(1).ReadAsArray()
    DEM[img_size] = preprocess_single(DEM[img_size], proc)

    raster = None

    return LSM[img_size], DEM[img_size]



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
                new_series[t] = preprocess_single(read_time(t, self.producer, analysis_time, print_filename=True), self.preprocess)

        self.data_series = new_series
        self.analysis_time = analysis_time

        return np.asarray(list(self.data_series.values()))
