import numpy as np
import sys
import datetime
import cv2
import os
from scipy import ndimage
from osgeo import gdal, osr
from base.fileutils import get_filename, gdal_read_from_http

DEM = {}
LSM = {}
GRID = None


def get_img_size(preprocess):
    for x in preprocess.split(","):
        k, v = x.split("=")
        if k == "img_size":
            return tuple(map(int, v.split("x")))
    return None


def reproject(arr, area):
    if area == "Scandinavia":
        return arr

    assert area == "SouthernFinland"

    ORIGINAL_EXTENT = [-1063327.181, 1338296.270, 1309172.819, -1334203.730]
    ORIGINAL_GEOTRANSFORM = [ORIGINAL_EXTENT[0], 2500, 0, ORIGINAL_EXTENT[1], 0, -2500]

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
    NEW_EXTENT = (
        [220000, 220000, 860000, -420000] if area == "SouthernFinland" else None
    )

    driver = gdal.GetDriverByName("MEM")
    arr_ds = driver.Create(
        "", xsize=arr.shape[1], ysize=arr.shape[0], bands=1, eType=gdal.GDT_Float32
    )

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ORIGINAL_WKT2)
    arr_ds.SetProjection(srs.ExportToWkt())
    arr_ds.SetGeoTransform(ORIGINAL_GEOTRANSFORM)

    band = arr_ds.GetRasterBand(1).WriteArray(np.squeeze(arr))

    ds = gdal.Translate("/vsimem/q", arr_ds, projWin=NEW_EXTENT)

    assert ds != None
    arr = ds.ReadAsArray()
    arr = np.expand_dims(arr, axis=2)
    ds = None
    return arr


def create_lonlat_grid(shape):
    """Create a longitude/latitude grid from MEPS2500D area"""

    x_len = 2370000
    y_len = 2670000

    dx = x_len / (shape[0] - 1)
    dy = y_len / (shape[1] - 1)

    src = osr.SpatialReference()
    tgt = osr.SpatialReference()
    src.ImportFromProj4(
        "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +x_0=1063327.1809 +y_0=1334203.7299 +ellps=WGS84 +units=m +no_defs"
    )
    tgt.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(src, tgt)
    coords = []
    # start bottom left
    for j in range(shape[1]):
        for i in range(shape[0]):
            p = transform.TransformPoint(i * dx, j * dy)
            lat = p[0]
            lon = p[1]

            assert lon > -20 and lon < 55
            assert lat > 45 and lat < 77

            coords.append((lon, lat))

    return np.flipud(np.asarray(coords).reshape(shape + (2,)))


def to_binary_mask(arr):
    arr[arr < 0.1] = 0.0
    arr[arr > 0] = 1.0

    return arr


def to_classes(arr, num_classes):
    return (np.around((100.0 * arr) / num_classes, decimals=0) * num_classes) / 100.0


def preprocess_many(imgs, process_label):
    return np.asarray(list(map(lambda x: preprocess_single(x, process_label), imgs)))


def preprocess_single(arr, process_label):
    for proc in process_label.split(","):
        k, v = proc.split("=")
        if k == "conv":
            v = int(v)
            kern = np.ones((v, v), np.float32) / (v * v)
            kern = np.expand_dims(kern, axis=2)
            arr = ndimage.convolve(arr, kern, mode="constant", cval=0.0)
        elif k == "to_binary_mask" and v == "true":
            arr = to_binary_mask(arr)
        elif k == "classes":
            arr = to_classes(arr, int(v))
        elif k == "standardize" and v == "true":
            arr = (arr - arr.mean()) / arr.std()
        elif k == "normalize" and v == "true":
            if np.min(arr) != np.max(arr):
                arr = (arr - np.min(arr)) / np.ptp(arr)
        elif k == "img_size":
            img_size = tuple(map(int, v.split("x")))
            if arr.shape != img_size:
                arr = np.expand_dims(
                    cv2.resize(arr, dsize=img_size, interpolation=cv2.INTER_LINEAR),
                    axis=2,
                )
        elif k == "area":  # and v != 'Scandinavia':
            arr = reproject(arr, v)

    return arr


def time_of_year_and_day(datetime):
    day = 24 * 60 * 60
    year = 365.2425 * day

    tod = np.sin(datetime.timestamp() * (2 * np.pi / day))
    toy = np.cos(datetime.timestamp() * (2 * np.pi / year))

    return tod, toy


def sun_declination_angle(ts):
    J = int(ts.strftime("%-j"))
    H = int(ts.strftime("%H"))
    M = int(ts.strftime("%M"))
    N = J + (H / 24.0) + (M / 60.0)
    decl = -23.44 * np.cos(np.deg2rad(360.0 / 365 * (N + 10)))

    return decl


def equation_of_time(ts):
    J = int(ts.strftime("%-j"))
    B = 360.0 * (J - 81) / 365
    B = np.deg2rad(B)
    E = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

    return E


def solar_hour_angle(ts, E, longitude):
    # need time zone to calculate solar hour
    # "normal" time zone cannot be used as it's a political
    # concept
    # use a straightforward longitude-based time zone
    dTZ = longitude / 15

    LSTM = 15 * dTZ
    TC = 4 * (25 - LSTM) + E  # minutes

    # local solar time
    LST = ts + datetime.timedelta(minutes=TC)
    LST = LST + datetime.timedelta(minutes=dTZ * 60)
    LST_H = int(LST.strftime("%H")) + int(LST.strftime("%M")) / 60.0

    # solar hour angle
    HRA = 15 * (LST_H - 12)
    return HRA


def sun_elevation_angle(decl, HRA, latitude):
    latitude = np.deg2rad(latitude)
    decl = np.deg2rad(decl)
    HRA = np.deg2rad(HRA)

    return np.rad2deg(
        np.arcsin(
            np.sin(latitude) * np.sin(decl)
            + np.cos(latitude) * np.cos(decl) * np.cos(HRA)
        )
    )


def create_sun_elevation_angle(ts, img_size):
    global GRID
    if GRID is None:
        GRID = create_lonlat_grid(img_size)

    decl = sun_declination_angle(ts)
    E = equation_of_time(ts)

    ret = []
    for c in GRID.reshape(img_size[0] * img_size[1], 2):
        HRA = solar_hour_angle(ts, E, c[0])
        angle = sun_elevation_angle(decl, HRA, c[1])
        ret.append(angle)

    ret = np.asarray(ret).reshape(img_size)
    ret = preprocess_single(ret, "normalize=true")
    ret = np.expand_dims(ret, axis=-1)
    return ret


def sun_elevation_angle_wrapper(ts, longitude, latitude):
    decl = sun_declination_angle(ts)
    E = equation_of_time(ts)
    HRA = solar_hour_angle(ts, E, longitude)
    angle = sun_elevation_angle(decl, HRA, latitude)

    return angle


def create_sun_elevation_angle_data(img_size):
    sun_file = get_filename(None, producer="cloudcast", param="sun_elevation_angle")

    print(f"Reading {sun_file}")

    if is_http(sun_file):
        import requests
        import io

        sun_file = sun_file.replace("s3://", "")
        sun_file = "https://lake.fmi.fi/{}".format(sun_file)
        response = requests.get(sun_file)
        response.raise_for_status()
        ds = np.load(io.BytesIO(response.content))
    else:
        ds = np.load(sun_file)

    datas = ds["arr_0"]
    times = ds["arr_1"]

    ret = {}
    for i, t in enumerate(times):
        ret[t] = datas[i]

    return ret


def create_datetime(datetime, img_size):
    tod, toy = time_of_year_and_day(datetime)
    tod = np.expand_dims(np.full(img_size, tod, dtype=np.float32), axis=-1)
    toy = np.expand_dims(np.full(img_size, toy, dtype=np.float32), axis=-1)

    return tod, toy


def process_lsm(LSM):
    # Value     Label
    # 11        Post-flooding or irrigated croplands (or aquatic)
    # 14        Rainfed croplands
    # 20        Mosaic cropland (50-70%) / vegetation (grassland/shrubland/forest) (20-50%)
    # 30        Mosaic vegetation (grassland/shrubland/forest) (50-70%) / cropland (20-50%)
    # 40        Closed to open (>15%) broadleaved evergreen or semi-deciduous forest (>5m)
    # 50        Closed (>40%) broadleaved deciduous forest (>5m)
    # 60        Open (15-40%) broadleaved deciduous forest/woodland (>5m)
    # 70        Closed (>40%) needleleaved evergreen forest (>5m)
    # 90        Open (15-40%) needleleaved deciduous or evergreen forest (>5m)
    # 100       Closed to open (>15%) mixed broadleaved and needleleaved forest (>5m)
    # 110       Mosaic forest or shrubland (50-70%) / grassland (20-50%)
    # 120       Mosaic grassland (50-70%) / forest or shrubland (20-50%)
    # 130       Closed to open (>15%) (broadleaved or needleleaved, evergreen or deciduous) shrubland (<5m)
    # 140       Closed to open (>15%) herbaceous vegetation (grassland, savannas or lichens/mosses)
    # 150       Sparse (<15%) vegetation
    # 160       Closed to open (>15%) broadleaved forest regularly flooded (semi-permanently or temporarily) - Fresh or brackish water
    # 170       Closed (>40%) broadleaved forest or shrubland permanently flooded - Saline or brackish water
    # 180       Closed to open (>15%) grassland or woody vegetation on regularly flooded or waterlogged soil - Fresh, brackish or saline water
    # 190       Artificial surfaces and associated areas (Urban areas >50%)
    # 200       Bare areas
    # 210       Water bodies
    # 220       Permanent snow and ice
    # 230       No data (burnt areas, clouds,)

    # forest
    LSM[np.logical_and(LSM >= 40, LSM <= 100)] = 0

    # urban
    LSM[LSM == 190] = 1

    # sparse / bare
    LSM[np.logical_or(LSM == 150, LSM == 200)] = 2

    # permanent snow
    LSM[LSM == 220] = 3

    # water
    LSM[LSM == 210] = 4

    # agriculture
    LSM[np.logical_and(LSM >= 11, LSM <= 30)] = 5

    # rest
    LSM[LSM > 5] = 6

    return LSM


def create_onehot_leadtime_conditioning(img_size, depth, active_layer):
    b = np.ones((1,) + img_size)
    return np.expand_dims(
        np.expand_dims(np.expand_dims(np.eye(depth)[active_layer], -1), 1) * b, axis=-1
    ).astype(np.short)


def create_squeezed_leadtime_conditioning(img_size, depth, active_leadtime):
    return np.expand_dims(
        np.full(img_size, active_leadtime / depth), axis=(0, 3)
    ).astype(np.float32)


def is_http(uri):
    return True if uri[0:4] == "http" or uri[0:5] == "s3://" else False


def create_topography_data(img_size):
    assert type(img_size) == tuple

    global DEM
    isize = "{}x{}".format(img_size[0], img_size[1])

    try:
        return DEM[isize]
    except KeyError as e:
        pass

    proc = "normalize=true"

    proc = "{},img_size={}".format(proc, isize)

    dem_file = get_filename(None, "DEM")

    print(f"Reading {dem_file}")

    raster = None

    if is_http(dem_file):
        raster = gdal_read_from_http(dem_file)
    else:
        raster = gdal.Open(dem_file)

    DEM[isize] = raster.GetRasterBand(1).ReadAsArray()
    DEM[isize] = preprocess_single(DEM[isize], proc).astype(np.float32)

    raster = None

    return DEM[isize]


def create_terrain_type_data(img_size):
    assert type(img_size) == tuple

    global LSM

    isize = "{}x{}".format(img_size[0], img_size[1])

    try:
        return LSM[isize]
    except KeyError as e:
        pass

    proc = "normalize=true"

    proc = "{},img_size={}".format(proc, isize)

    lsm_file = get_filename(None, "LSM")

    print(f"Reading {lsm_file}")

    if is_http(lsm_file):
        raster = gdal_read_from_http(lsm_file)
    else:
        raster = gdal.Open(lsm_file)

    LSM[isize] = raster.GetRasterBand(1).ReadAsArray()
    LSM[isize] = process_lsm(LSM[isize])
    LSM[isize] = preprocess_single(LSM[isize], proc).astype(np.float32)

    raster = None

    return LSM[isize]


def generate_clim_values(shape, month):
    assert month >= 1 and month <= 12
    month -= 1

    d = np.load(get_filename(None, "clim"))
    m = d["arr_0"][month]
    x = m[..., 0]
    y = m[..., 1]

    lst = []

    for _ in range(shape[0]):
        lst.append(np.random.choice(y, size=(shape[1], shape[2]), p=x))

    return np.asarray(lst)
