import numpy as np
import glob
import sys
import eccodes as ecc
import datetime
#import tensorflow as tf
import cv2
import os
from scipy import ndimage
from PIL import Image
from tensorflow import keras


INPUT_DIR = '/home/partio/cloudnwc/effective_cloudiness/data/'

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

        ecc.codes_release(gh)
        return data


def save_grib(data, filepath, datetime):
    assert(filepath[-5:] == 'grib2')

    try:
        os.makedirs(os.path.dirname(outfile))
    except FileExistsError as e:
        pass

    data = np.flipud(data)

    with open(outfile) as fp:
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "grid_type", "lambert")
        ecc.codes_set(h, 'shapeOfTheEarth', 5)
        ecc.codes_set(h, 'Nx', 949)
        ecc.codes_set(h, 'Ny', 1069)
        ecc.codes_set(h, 'DxInMeters', 2500)
        ecc.codes_set(h, 'DyInMeters', 2500)
        ecc.codes_set(h, 'jScansPositive', 1)
        ecc.codes_set(h, "latitudeOfFirstPointInDegrees", 50.3196)
        ecc.codes_set(h, "longitudeOfFirstPointDegrees", 0.27828)
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

        with open(outfile, 'wb') as fpout:
            ecc.codes_write(h, fpout)
            print(f'Wrote file {outfile}')

        ecc.codes_release(h)


def read_gribs(filenames):

    def process_grib(file_path):
        img = read_grib(file_path)
        return img

    files_ds = []

    i = 0
    for f in filenames:
        i = i + 1

        files_ds.append(process_grib(f))

    if len(files_ds) == 0:
        print("No files found")

    return np.asarray(files_ds)