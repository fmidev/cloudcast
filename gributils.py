import numpy as np
import eccodes as ecc
import datetime
import cv2
import os
import requests
import sys
import boto3
from io import BytesIO

DEFAULT_SIZE=(1069, 949, 1)

def write_to_s3(object_name, data, **kwargs):
    s3 = boto3.resource('s3',
      endpoint_url = 'https://{}'.format(os.environ['S3_HOSTNAME']),
      aws_access_key_id = os.environ['S3_ACCESS_KEY_ID'],
      aws_secret_access_key = os.environ['S3_SECRET_ACCESS_KEY']
    )

    bucket_name = object_name[5:].split('/')[0]
    obj_name = '/'.join(object_name[5:].split('/')[1:])

    bucket = s3.Bucket(bucket_name)

    data.seek(0)

    bucket.upload_fileobj(data, obj_name)


def read_from_http(url, **kwargs):
    r = requests.get(url, stream=True)

    if r.status_code == 404:
        print(f"Not found: {url}")
        return np.full(DEFAULT_SIZE, np.NAN)
    if r.status_code != 200:
        print(f'HTTP error: {r.status_code}')
        sys.exit(1)

    gh = ecc.codes_new_from_message(r.content)
    return read_grib_contents(gh, fileuri=url, **kwargs)


def read_from_file(file_path, message_no, **kwargs):
    try:
        with open(file_path) as fp:
            gh = ecc.codes_new_from_file(fp, ecc.CODES_PRODUCT_GRIB)
            return read_grib_contents(gh, fileuri=file_path, **kwargs)
    except FileNotFoundError as e:
        print(e)
        return np.full(DEFAULT_SIZE, np.NAN)


def read_grib_contents(gh, **kwargs):
    ni = ecc.codes_get_long(gh, "Ni")
    nj = ecc.codes_get_long(gh, "Nj")

    dtype = kwargs.get('dtype')
    data = ecc.codes_get_double_array(gh, "values").astype(dtype).reshape(nj, ni)

    img_size = kwargs.get('img_size', None)
    if img_size is not None:
        data = np.expand_dims(cv2.resize(data, dsize=img_size, interpolation=cv2.INTER_LINEAR), axis=2)

    if ecc.codes_get(gh, "jScansPositively"):
        data = np.flipud(data) # image data is +x-y

    ecc.codes_release(gh)

    if np.max(data) == 9999.0 and np.min(data) == 9999.0:
        data[data == 9999.0] = np.NAN
        return np.expand_dims(data, axis=-1)

    if dtype != np.uint8 and np.max(data) > 1.1:
        data = data / 100.0

    if np.min(data) < -0.01 or np.max(data) > 1.01:
        print("Invalid data found from '{}': min={} max={}".format(kwargs['fileuri'], np.min(data), np.max(data)))
        sys.exit(1)

    data = np.expand_dims(data, axis=2)

    return data


def read_grib(file_path, message_no = 0, **kwargs):
    print_filename=kwargs.get('print_filename', False)

    if print_filename:
        print(f"Reading {file_path}")

    if file_path[0:4] == 'http':
        return read_from_http(file_path, **kwargs)
    else:
        return read_from_file(file_path, message_no, **kwargs)


def save_grib(data, filepath, analysistime, forecasttime):
    assert(filepath[-5:] == 'grib2')

    number = 164 if filepath.find('unet') != -1 else 192

    h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ecc.codes_set(h, "gridType", "lambert")
    ecc.codes_set(h, 'shapeOfTheEarth', 5)
    ecc.codes_set(h, 'Nx', data.shape[1])
    ecc.codes_set(h, 'Ny', data.shape[0])
    ecc.codes_set(h, 'DxInMetres', 2370000 / (data.shape[1]-1))
    ecc.codes_set(h, 'DyInMetres', 2670000 / (data.shape[0]-1))
    ecc.codes_set(h, 'jScansPositively', 1)
    ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
    ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
    ecc.codes_set(h, "Latin1InDegrees", 63.3)
    ecc.codes_set(h, "Latin2InDegrees", 63.3)
    ecc.codes_set(h, "LoVInDegrees", 15)
    ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
    ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
    ecc.codes_set(h, "dataDate", int(analysistime.strftime('%Y%m%d')))
    ecc.codes_set(h, "dataTime", int(analysistime.strftime('%H%M')))
    ecc.codes_set(h, "centre", 86)
    ecc.codes_set(h, "generatingProcessIdentifier", 251)
    ecc.codes_set(h, "discipline", 192)
    ecc.codes_set(h, "parameterCategory", 128)
    ecc.codes_set(h, "parameterNumber", number)
    ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
    ecc.codes_set(h, "packingType", "grid_ccsds")
    ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 0)
    ecc.codes_set(h, "forecastTime", int((forecasttime - analysistime).total_seconds()/60))

    data = np.flipud(data)
    ecc.codes_set_values(h, data.flatten())

    if filepath[0:5] == 's3://':
        bio = BytesIO()
        ecc.codes_write(h, bio)

        write_to_s3(filepath, bio)
    else:
        try:
            os.makedirs(os.path.dirname(filepath))
        except FileExistsError as e:
            pass

        with open(filepath, 'wb') as fp:
            ecc.codes_write(h, fp)

    print(f'Wrote file {filepath}')

    ecc.codes_release(h)


def read_gribs(filenames, **kwargs):

    files_ds = []

    for f in filenames:
        files_ds.append(read_grib(f, **kwargs))
    if len(files_ds) == 0:
        print("No files found")

    files_ds = np.asarray(files_ds)

    if files_ds.shape[-2] == 1:
        files_ds = np.squeeze(files_ds, -2)

    return files_ds
