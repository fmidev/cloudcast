import numpy as np
import sys
import datetime
import cv2
import os
from scipy import ndimage
from PIL import Image, ImageEnhance
from osgeo import gdal,osr


def get_img_size(preprocess):
    for x in preprocess.split(','):
        k,v = x.split('=')
        if k == 'img_size':
            return tuple(map(int, v.split('x')))
    return None


def reproject(arr, area):
    if area == 'Scandinavia':
        return arr

    assert(area == 'SouthernFinland')

    ORIGINAL_EXTENT = [-1063327.181, 1338296.270, 1309172.819, -1334203.730 ]
    ORIGINAL_GEOTRANSFORM = [ ORIGINAL_EXTENT[0], 2500, 0, ORIGINAL_EXTENT[1], 0, -2500 ]

    ORIGINAL_WKT2 = """
PROJCRS["unknown",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4326]],
    CONVERSION["Lambert Conic Conformal (2SP)",
        METHOD["Lambert Conic Conformal (2SP)",
            ID["EPSG",9802]],
        PARAMETER["Latitude of false origin",63.3,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8821]],
        PARAMETER["Longitude of false origin",15,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8822]],
        PARAMETER["Latitude of 1st standard parallel",63.3,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Latitude of 2nd standard parallel",63.3,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8824]],
        PARAMETER["Easting at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8826]],
        PARAMETER["Northing at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8827]]],
    CS[Cartesian,2],
        AXIS["easting",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["northing",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]
    """
    NEW_EXTENT = [220000, 220000, 860000, -420000] if area == 'SouthernFinland' else None

    driver = gdal.GetDriverByName('MEM')
    arr_ds = driver.Create('', xsize=arr.shape[1], ysize=arr.shape[0], bands=1, eType=gdal.GDT_Float32)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ORIGINAL_WKT2)
    arr_ds.SetProjection(srs.ExportToWkt())
    arr_ds.SetGeoTransform(ORIGINAL_GEOTRANSFORM)

    band = arr_ds.GetRasterBand(1).WriteArray(np.squeeze(arr))

    ds = gdal.Translate('/vsimem/q', arr_ds, projWin = NEW_EXTENT)

    assert(ds != None)
    arr = ds.ReadAsArray()
    arr = np.expand_dims(arr, axis=2)
    ds = None
    return arr

def to_binary_mask(arr):
    arr[arr < 0.1] = 0.0
    arr[arr > 0] = 1.0

    return arr

def to_classes(arr, num_classes):
    return (np.around((100.0 * arr) / num_classes, decimals=0) * num_classes) / 100.0

def preprocess_many(imgs, process_label):
    return np.asarray(list(map(lambda x: preprocess_single(x, process_label), imgs)))


def preprocess_single(arr, process_label):
    for proc in process_label.split(','):
        k,v = proc.split('=')
        if k == 'conv':
            v = int(v)
            kern = np.ones((v,v), np.float32) / (v * v)
            kern = np.expand_dims(kern, axis=2)
            arr = ndimage.convolve(arr, kern, mode='constant', cval=0.0)
        elif k == 'to_binary_mask' and v == 'true':
            arr = to_binary_mask(arr)
        elif k == 'classes':
            arr = to_classes(arr, int(v))
        elif k == 'standardize' and v == 'true':
            arr = (arr - arr.mean()) / arr.std()
        elif k == 'img_size':
            img_size = tuple(map(int, v.split('x')))
            arr = np.expand_dims(cv2.resize(arr, dsize=img_size, interpolation=cv2.INTER_LINEAR), axis=2)
        elif k == 'area': # and v != 'Scandinavia':
            arr = reproject(arr, v)

    return arr


def sharpen(data, factor):
#    assert(data.shape == (1,) + IMG_SIZE + (1,))
    im = Image.fromarray(np.squeeze(data) * 255)
    im = im.convert('L')

    enhancer = ImageEnhance.Sharpness(im)
    sharp = np.array(enhancer.enhance(factor)) / 255.0
    return np.expand_dims(sharp, [0,3])


def time_of_year_and_day(datetime):
    day = 24*60*60
    year = 365.2425 * day

    tod = np.sin(datetime.timestamp() * (2 * np.pi / day))
    toy = np.cos(datetime.timestamp() * (2 * np.pi / year))

    return tod, toy
