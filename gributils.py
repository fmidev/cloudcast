import numpy as np
import eccodes as ecc
import datetime
import cv2
import os
import requests
import sys

DEFAULT_SIZE=(1069, 949, 1)

def read_from_http(url):
    r = requests.get(url, stream=True)

    if r.status_code == 404:
        print(f"Not found: {url}")
        return np.full(DEFAULT_SIZE, np.NAN)
    if r.status_code != 200:
        print(f'HTTP error: {r.status_code}')
        sys.exit(1)

    gh = ecc.codes_new_from_message(r.content)
    return read_grib_contents(gh)


def read_from_file(file_path, message_no):
    try:
        with open(file_path) as fp:
            gh = ecc.codes_new_from_file(fp, ecc.CODES_PRODUCT_GRIB)
            return read_grib_contents(gh)
    except FileNotFoundError as e:
        print(e)
        return np.full(DEFAULT_SIZE, np.NAN)


def read_grib_contents(gh):
    ni = ecc.codes_get_long(gh, "Ni")
    nj = ecc.codes_get_long(gh, "Nj")

    data = ecc.codes_get_double_array(gh, "values").astype(np.float32).reshape(nj, ni)

    if ecc.codes_get(gh, "jScansPositively"):
        data = np.flipud(data) # image data is +x-y

    ecc.codes_release(gh)

    if np.max(data) == 9999.0 and np.min(data) == 9999.0:
        data[data == 9999.0] = np.NAN
        return np.expand_dims(data, axis=-1)

    if np.max(data) > 1.1:
        data = data / 100.0

    data = np.expand_dims(data, axis=2)

    return data


def read_grib(file_path, message_no = 0, **kwargs):
    print_filename=kwargs.get('print_filename', False)

    if print_filename:
        print(f"Reading {file_path}")

    if file_path[0:4] == 'http':
        return read_from_http(file_path)
    else:
        return read_from_file(file_path, message_no)


def save_grib(data, filepath, analysistime, forecasttime):
    assert(filepath[-5:] == 'grib2')

    try:
        os.makedirs(os.path.dirname(filepath))
    except FileExistsError as e:
        pass



    h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ecc.codes_set(h, "gridType", "lambert")
    ecc.codes_set(h, 'shapeOfTheEarth', 5)
    ecc.codes_set(h, 'Nx', data.shape[0])
    ecc.codes_set(h, 'Ny', data.shape[1])
    ecc.codes_set(h, 'DxInMetres', 2372500 / data.shape[0])
    ecc.codes_set(h, 'DyInMetres', 2672500 / data.shape[1])

#        ecc.codes_set(h, 'Nx', 949)
#        ecc.codes_set(h, 'Ny', 1069)
#        ecc.codes_set(h, 'DxInMeters', 2500)
#        ecc.codes_set(h, 'DyInMeters', 2500)
    ecc.codes_set(h, 'jScansPositively', 1)
    ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.3196)
    ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
    ecc.codes_set(h, "Latin1InDegrees", 63.3)
    ecc.codes_set(h, "Latin2InDegrees", 63.3)
    ecc.codes_set(h, "LoVInDegrees", 15)
    ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
    ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
    ecc.codes_set(h, "dataDate", int(analysistime.strftime('%Y%m%d')))
    ecc.codes_set(h, "dataTime", int(analysistime.strftime('%H%M')))
    ecc.codes_set(h, "centre", 86)
    ecc.codes_set(h, "generatingProcessIdentifier", 255)
    ecc.codes_set(h, "discipline", 192)
    ecc.codes_set(h, "parameterCategory", 128)
    ecc.codes_set(h, "parameterNumber", 164)
    ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
    ecc.codes_set(h, "packingType", "grid_ccsds")
    ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 0)
    ecc.codes_set(h, "forecastTime", int((forecasttime - analysistime).total_seconds()/60))

    data = np.flipud(data)

    ecc.codes_set_values(h, data.flatten())

    with open(filepath, 'wb') as fp:
        ecc.codes_write(h, fp)
        print(f'Wrote file {filepath}')

    ecc.codes_release(h)


def read_gribs(filenames):

    files_ds = []

    i = 0
    for f in filenames:
        i = i + 1

        files_ds.append(read_grib(f))
    if len(files_ds) == 0:
        print("No files found")

    return np.asarray(files_ds)
