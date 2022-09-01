import qnorm
import numpy as np
from osgeo import gdal, osr

def quantile_normalize(raw, ref):
    assert(raw.shape == ref.shape)
    shape = raw.shape

    raw = np.expand_dims(raw.flatten().transpose(), axis=0).transpose()
    ref = ref.flatten()
    q = qnorm.quantile_normalize(raw, target=ref)
    return q.reshape(shape)


def downscale(arr, new_size):
    srs = '+proj=lcc +lat_0=0 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +units=m +no_defs +ellps=WGS84 +datum=WGS84 +x_0=0 +y_0=0'

    sr = osr.SpatialReference()
    sr.ImportFromProj4(srs)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # adfGeoTransform[0] /* top left x */
    # adfGeoTransform[1] /* w-e pixel resolution */
    # adfGeoTransform[2] /* 0 */
    # adfGeoTransform[3] /* top left y */
    # adfGeoTransform[4] /* 0 */
    # adfGeoTransform[5] /* n-s pixel resolution (negative value) */

    si = 2370000
    sj = 2670000

    di = si / (arr.shape[1] - 1)
    dj = -(sj / (arr.shape[0] - 1))

    gt = ( -1061372.160, di, 0, 1338684.199, 0, dj)

    minx = gt[0]
    maxy = gt[3]
    maxx = minx + si
    miny = maxy - sj

    extent = [ minx, miny, maxx, maxy ]
    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('/vsimem/y', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    b = ds.GetRasterBand(1)
    b.WriteArray(arr)
    b.FlushCache()

    ds.SetSpatialRef(sr)
    ds.SetGeoTransform(gt)

    scaled = gdal.Warp('',
            ds,
            format='VRT',
            width=new_size[1],
            height=new_size[0],
            outputBounds=extent,
            resampleAlg=gdal.GRIORA_Bilinear)

    narr = scaled.GetRasterBand(1).ReadAsArray()
    scaled = None

    return narr

