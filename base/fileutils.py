import numpy as np
import glob
import sys
import datetime
import os
import requests
from base.gributils import *
from base.s3utils import *
from osgeo import gdal, osr

try:
    INPUT_DIR = os.environ["CLOUDCAST_INPUT_DIR"]
except KeyError:
    INPUT_DIR = "s3://cc_archive"


def read_filenames(
    start_time, stop_time, producer="nwcsaf", param="effective-cloudiness"
):
    print(f"Input directory: {INPUT_DIR}/{producer}")

    if INPUT_DIR.startswith("http") or INPUT_DIR.startswith("s3://"):
        return read_filenames_from_s3(start_time, stop_time, producer)

    files = sorted(
        glob.glob(f"{INPUT_DIR}/{producer}/**/*{param}*.grib2", recursive=True)
    )

    start_date = int(start_time.strftime("%Y%m%d"))
    stop_date = int(stop_time.strftime("%Y%m%d"))
    filenames = []

    for f in files:
        datetime = int(f.split("/")[-1][0:8])
        if datetime >= start_date and datetime < stop_date:
            filenames.append(f)

    print("Filter matched {} files".format(len(filenames)))
    return filenames


def get_filename(
    time, producer="nwcsaf", analysis_time=None, param="effective-cloudiness"
):
    if producer == "nwcsaf":
        return "{}/nwcsaf/{}_nwcsaf_{}.grib2".format(
            INPUT_DIR, time.strftime("%Y/%m/%d/%Y%m%dT%H%M%S"), param
        )
    if producer == "mnwc":
        if analysis_time is None:
            # return newest possible
            return "{}/mnwc/{}.grib2".format(
                INPUT_DIR, time.strftime("%Y%m%d%H00+000h%Mm")
            )
        else:
            lt = time - analysis_time
            lt_h = int(lt.total_seconds() // 3600)
            lt_m = int(lt.total_seconds() // 60 % 60)
            return "{}/mnwc/{}00+{:03d}h{:02d}m.grib2".format(
                INPUT_DIR, analysis_time.strftime("%Y%m%d%H"), lt_h, lt_m
            )
    if producer == "meps":
        assert analysis_time is not None
        ahour = int(analysis_time.strftime("%H"))
        ahour = ahour - ahour % 3
        lt = time - analysis_time
        lt_h = int(lt.total_seconds() // 3600)
        lt_m = int(lt.total_seconds() // 60 % 60)
        return "{}/meps/{}/{}{:02d}00+{:03d}h{:02d}m-{}.grib2".format(
            INPUT_DIR,
            analysis_time.strftime("%Y/%m/%d"),
            analysis_time.strftime("%Y%m%d"),
            ahour,
            lt_h,
            lt_m,
            param,
        )
    if producer == "DEM":
        return "{}/static/DEM-cloudcast.tif".format(INPUT_DIR)
    if producer == "LSM":
        return "{}/static/LSM-cloudcast.tif".format(INPUT_DIR)
    if producer == "clim":
        return "{}/static/climatology-monthly-128x128.npz".format(INPUT_DIR)


def read_time(
    time, producer="nwcsaf", analysis_time=None, param="effective-cloudiness", **kwargs
):
    return read_grib(get_filename(time, producer, analysis_time, param), **kwargs)


def read_times(times, producer="nwcsaf", analysis_time=None):
    data = []
    for time in times:
        try:
            data.append(read_grib(get_filename(time, producer, analysis_time)))
        except FileNotFoundError as e:
            pass

    return data


def gdal_read_from_http(url):
    mmap_name = "/vsimem/xxx"

    def s3_to_http(url):
        if url[0:5] == "s3://":
            tokens = url[5:].split("/")
            tokens[0] = "{}/{}".format(os.environ["S3_HOSTNAME"], tokens[0])
            url = "https://" + "/".join(tokens)
        return url

    def read_from_http(url):
        r = requests.get(s3_to_http(url), stream=True)

        if r.status_code != 200:
            print(f"Not found: {url}")
            sys.exit(1)
        return r.content

    gdal.FileFromMemBuffer(mmap_name, read_from_http(url))
    return gdal.Open(mmap_name)
