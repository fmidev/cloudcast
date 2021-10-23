import numpy as np
import glob
import sys
import eccodes as ecc
import datetime
#import tensorflow as tf
import cv2
from scipy import ndimage
from PIL import Image
from tensorflow import keras


INPUT_DIR = '/home/partio/cloudnwc/effective_cloudiness/data/grib2'


def preprocess_many(imgs, img_size):
    ds = []

    for img in imgs:
        ds.append(preprocess_single(img, img_size))

    return np.asarray(ds)


def preprocess_single(img, img_size, kernel_size = (3,3), num_classes = 10):

    kernel1 = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    kernel1 = np.expand_dims(kernel1, axis=2)

#    print(f"Preprocessing with {kernel_size} kernel and rounding to {num_classes} classes")

    ds = []

    conv = ndimage.convolve(img, kernel1, mode='constant', cval=0.0)
    conv = (np.around((100 * conv) / num_classes, decimals=0)*num_classes)/100

#        im = Image.fromarray(img.squeeze() * 255)
#        im = im.convert('L')
#        im.save("original.jpg")

#        im = Image.fromarray(conv.squeeze() * 255)
#        im = im.convert('L')
#        im.save("convoluted.jpg")

#        print(img)
#        print(conv)
#        print(np.histogram(img))
#        print(np.histogram(conv))
#        break

    if img_size is not None:
        conv = np.expand_dims(cv2.resize(conv, dsize=img_size, interpolation=cv2.INTER_LINEAR), axis=2)

 
    return conv #.astype(np.float16)


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


def read_filenames(start_time, stop_time):
    print(f'Input directory: {INPUT_DIR}')

    files = sorted(glob.glob(f'{INPUT_DIR}/**/*.grib2', recursive=True))

    start_date = start_time.strftime("%Y%m%d")
    stop_date = stop_time.strftime("%Y%m%d")

    filenames = []

    for f in files:
        datetime = f.split('/')[-1][0:14]
        if datetime >= start_date and datetime < stop_date:
            filenames.append(f)

    return filenames

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


def sharpen(data, factor):
    assert(data.shape == (1,) + IMG_SIZE + (1,))
    im = Image.fromarray(np.squeeze(data) * 255)
    im = im.convert('L')

    enhancer = ImageEnhance.Sharpness(im)
    sharp = np.array(enhancer.enhance(factor)) / 255.0
    return np.expand_dims(sharp, [0,3])


def read_time(time):
    return read_grib('{}/{}_nwcsaf_effective-cloudiness.grib2'.format(INPUT_DIR, time.strftime('%Y/%m/%d/%Y%m%dT%H%M%S')))


def create_train_val_split_timeseries(dataset):

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y


    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return (x_train, y_train, x_val, y_val)


def create_train_val_split(dataset, train_history_len=1):

    assert(train_history_len is not None)

    if dataset.shape[0] % 2 == 1:
        dataset = dataset[:-1]

    n_split = dataset.shape[0] / (train_history_len + 1)
    dataset = np.asarray(np.split(dataset, n_split))

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def split_to_train_test(data):
        x = data[:, 0 : data.shape[1] - 1, :, :].squeeze(1)
        y = data[:, -1, :, :]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = split_to_train_test(train_dataset)
    x_val, y_val = split_to_train_test(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return (x_train, y_train, x_val, y_val)


def create_dataset(start_time, stop_time, img_size=None, preprocess=False):
    print(f"Creating dataset with time range {start_time} to {stop_time}")

    filenames = read_filenames(start_date, stop_date)
    ds = read_gribs(filenames)

    print(f"Dataset shape: {ds.shape}")

    if preprocess:
        ds = preprocess_many(ds, img_size)
        print(f'Dataset shape: {ds.shape}')

    return ds




class EffectiveCloudinessGenerator(keras.utils.Sequence):

    def __init__(self, start_date, stop_date, n_channels=1, batch_size=32, img_size=(256,256)):
        self.filenames = read_filenames(start_date, stop_date)
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.img_size = img_size
        self.initial = True

        print(f"Number of files: {len(self.filenames)}")

    def __len__(self):
        return (np.floor(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        i = -1

        while i < self.batch_size:
            for j in range(self.n_channels):
                batch_x.append(self.filenames[idx * self.batch_size + i + 1])
                i += 1
            batch_y.append(self.filenames[idx * self.batch_size + i + 1])
            i += 1

        x = preprocess_many(read_gribs(batch_x), self.img_size)
        y = preprocess_many(read_gribs(batch_y), self.img_size)

        if self.initial:
            print(f'Training batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y

