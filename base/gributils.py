import numpy as np
import eccodes as ecc
import datetime
import cv2
import os
import requests
import sys
import time

# import tensorflow as tf
from io import BytesIO
from base.s3utils import *

DEFAULT_SIZE = (1069, 949, 1)
GRIB_CACHE = {}


def read_from_http(url, **kwargs):
    if url[0:5] == "s3://":
        tokens = url[5:].split("/")
        tokens[0] = "{}/{}".format(os.environ["S3_HOSTNAME"], tokens[0])
        url = "https://" + "/".join(tokens)

    r = requests.get(url, stream=True)
    if r.status_code == 404:
        print(f"Not found: {url}")
        return np.full(DEFAULT_SIZE, np.NAN)
    if r.status_code != 200:
        print(f"HTTP error: {r.status_code}")
        sys.exit(1)

    print_filename = kwargs.get("print_filename", False)

    if print_filename:
        print(f"Reading {url}")

    gh = ecc.codes_new_from_message(r.content)
    return read_grib_contents(gh, fileuri=url, **kwargs)


def read_from_file(file_path, message_no, **kwargs):
    try:
        with open(file_path) as fp:
            print_filename = kwargs.get("print_filename", False)

            if print_filename:
                print(f"Reading {file_path}")

            gh = ecc.codes_new_from_file(fp, ecc.CODES_PRODUCT_GRIB)
            return read_grib_contents(gh, fileuri=file_path, **kwargs)
    except FileNotFoundError as e:
        print(e)
        return np.full(DEFAULT_SIZE, np.NAN)


def read_grib_contents(gh, **kwargs):
    ni = ecc.codes_get_long(gh, "Ni")
    nj = ecc.codes_get_long(gh, "Nj")
    cat = ecc.codes_get_long(gh, "parameterCategory")
    num = ecc.codes_get_long(gh, "parameterNumber")

    dtype = kwargs.get("dtype", np.single)

    data = ecc.codes_get_double_array(gh, "values").astype(dtype).reshape(nj, ni)

    disable_preprocess = kwargs.get("disable_preprocess", False)

    if disable_preprocess:
        ecc.codes_release(gh)
        return np.expand_dims(data, axis=2)

    img_size = kwargs.get("img_size", None)
    if img_size is not None:
        data = np.expand_dims(
            cv2.resize(data, dsize=img_size, interpolation=cv2.INTER_LINEAR), axis=2
        )
        # data = tf.image.resize(np.expand_dims(data, -1), img_size).numpy().astype(dtype)
    if ecc.codes_get(gh, "jScansPositively"):
        data = np.flipud(data)  # image data is +x-y

    ecc.codes_release(gh)

    if np.max(data) == 9999.0:  # and np.min(data) == 9999.0:
        data[data == 9999.0] = np.NAN
        return np.expand_dims(data, axis=-1)

    if dtype != np.uint8 and ((cat == 6 and num == 1) or (cat == 6 and num == 192)):
        data = data / 100.0

    return np.expand_dims(data, axis=2)


def read_grib(file_path, message_no=0, **kwargs):
    global GRIB_CACHE

    enable_cache = kwargs.get("enable_cache", False)

    if enable_cache:
        try:
            return GRIB_CACHE[file_path]
        except KeyError:
            pass

    if file_path[0:4] == "http" or file_path[0:5] == "s3://":
        arr = read_from_http(file_path, **kwargs)
    else:
        arr = read_from_file(file_path, message_no, **kwargs)

    if enable_cache:
        GRIB_CACHE[file_path] = arr

    return arr


def save_grib(datas, filepath, analysistime, forecasttimes, grib_options=None):
    assert filepath[-5:] == "grib2"
    assert len(datas) == len(forecasttimes)

    bio = BytesIO()

    for data, forecasttime in zip(datas, forecasttimes):
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "gridType", "lambert")
        ecc.codes_set(h, "shapeOfTheEarth", 6)
        ecc.codes_set(h, "Nx", data.shape[1])
        ecc.codes_set(h, "Ny", data.shape[0])
        ecc.codes_set(h, "DxInMetres", 2370000 / (data.shape[1] - 1))
        ecc.codes_set(h, "DyInMetres", 2670000 / (data.shape[0] - 1))
        ecc.codes_set(h, "jScansPositively", 1)
        ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
        ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
        ecc.codes_set(h, "Latin1InDegrees", 63.3)
        ecc.codes_set(h, "Latin2InDegrees", 63.3)
        ecc.codes_set(h, "LoVInDegrees", 15)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", 251)
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", 6)
        ecc.codes_set(h, "parameterNumber", 192)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 0)
        ecc.codes_set(
            h, "forecastTime", int((forecasttime - analysistime).total_seconds() / 60)
        )
        ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
        ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products

        if grib_options is not None:
            for gopt in grib_options.split(","):
                k, v = gopt.split("=")
                typ = "d"
                elem = k.split(":")
                if len(elem) == 2:
                    typ = elem[1]
                if typ == "d":
                    v = int(v)
                elif typ == "f":
                    v = float(v)

                ecc.codes_set(h, k, v)

        data = np.flipud(data)
        ecc.codes_set_values(h, data.flatten())
        ecc.codes_write(h, bio)
        ecc.codes_release(h)

    if filepath[0:5] == "s3://":
        write_to_s3(filepath, bio)
    else:
        try:
            os.makedirs(os.path.dirname(filepath))
        except FileExistsError as e:
            pass

        with open(filepath, "wb") as fp:
            fp.write(bio.getbuffer())

    print(f"Wrote file {filepath}")


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
