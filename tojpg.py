import eccodes as ecc
import numpy as np
import glob
import os

from PIL import Image

def convert_to_jpeg(filename):

    outdir = '/home/partio/tmp/cloudnwc-jpeg/raw'
    try:
        os.makedirs(outdir)
    except FileExistsError as e:
        pass

    with open(filename) as fp:

        gh = ecc.codes_new_from_file(fp, ecc.CODES_PRODUCT_GRIB)
        year = ecc.codes_get(gh, "year")
        month = ecc.codes_get(gh, "month")
        day = ecc.codes_get(gh, "day")
        hour = ecc.codes_get(gh, "hour")
        minute = ecc.codes_get(gh, "minute")

        date = f"{year}{month:02d}{day:02d}T{hour:02d}{minute:02d}"
        outfile = f'{outdir}/{date}.jpg'
        if os.path.exists(outfile):
            return

        ni = ecc.codes_get_long(gh, "Ni")
        nj = ecc.codes_get_long(gh, "Nj")

        data = np.asarray(ecc.codes_get_double_array(gh, "values")).reshape(nj, ni)
        data = data / 100.0 * 255
        data = np.flipud(data)

        #print(data.shape)
        #print(data)
        im = Image.fromarray(data)
        #im.show()
        im = im.convert('L')
        im.save(outfile)
        print(f'Wrote file {outfile}')


def convert(year,month):
    indir = f'/home/partio/cloudnwc/effective_cloudiness/data/grib2/{year}/{month:02d}/'

    for f in glob.glob(f'{indir}/*/*grib2'):
        convert_to_jpeg(f)


convert(2020,10)
