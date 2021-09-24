#import matplotlib
#matplotlib.use("TkAgg")

import requests
import eccodes as ecc
import numpy as np
import pathlib
import glob

from tensorflow.keras.models import save_model # import datasets, models, layers
from PIL import Image
from model import *

INPUT_BUCKET = "nwcsaf-effective-cloudiness-data"
IMG_SIZE = (512,512)

def read_input(year, months):
    input = { 'date' : [], 'data' : [] }

    for month in months:
        for day in range(1,2):
            for hour in range(0,1):
                for min in ("00", "15", "30", "45"):
                    url = f'https://nwcsaf-effective-cloudiness-data.lake.fmi.fi/grib2/{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}T{hour:02d}{min}00_nwcsaf_effective-cloudiness.grib2'
                    print(url)
                    response = requests.get(url, proxies={"https":"http://wwwcache.fmi.fi:8080"})
                    if response.status_code == 200:
                        gh = ecc.codes_new_from_message(response.content)

                        date = "{}{}".format(ecc.codes_get(gh, "dataDate"), ecc.codes_get(gh, "dataTime"))
                        data = ecc.codes_get_double_array(gh, "values")
                        ni = ecc.codes_get_long(gh, "Ni")
                        nj = ecc.codes_get_long(gh, "Nj")

                        input['date'].append(date)
                        input['data'].append(np.asarray(data).reshape(ni,nj))

    return input

def convert_to_jpeg(filename):
    outdir = '/home/partio/tmp/cloudnwc-jpeg'
#    os.makedirs(outdir)
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

def read_input_disk(year, months, type):
   
#    input_dir = f'/home/partio/cloudnwc/effective_cloudiness/data/grib2/{year}/{months[0]:02d}/01/'

#    for f in glob.glob(f'{input_dir}/*grib2'):
#        convert_to_jpeg(f)

    input_dir = pathlib.Path(f'/home/partio/tmp/cloudnwc-jpeg/')
    print(input_dir)

    x_files = glob.glob(f'{input_dir}/train/*.jpg')
    y_files = glob.glob(f'{input_dir}/test/*.jpg')

    files_ds = tf.data.Dataset.from_tensor_slices((x_files, y_files))


    def process_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=IMG_SIZE)
        return img

    files_ds = files_ds.map(lambda x, y: (process_img(x), process_img(y))).batch(1)

    return files_ds

#    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#      input_dir,
#      labels=None,
#      color_mode='grayscale',
#      seed=123,
#      image_size=(256,256),
#      label_mode=None,
#      batch_size=16)
#
#    return train_ds



train_ds = read_input_disk(2021, [1], 'train')
#test_ds = read_input_disk(2021, [1], 'eval')

#print(train_ds.take(1))
#print(test_ds.take(1))

#train_x, train_y = train_ds.load()
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

m = unet(input_size=IMG_SIZE + (1,))
#model = model()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/cp.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)

hist = m.fit(train_ds, epochs=10, batch_size=15, callbacks=[cp_callback])


save_model(m, './cloudcast.model')
#model_checkpoint = ModelCheckpoint('ccast.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])



#model = vanilla_unet(input_shape=(512, 512, 3))

#print(model)

