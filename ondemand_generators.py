import datetime
import numpy as np
from tensorflow import keras
from fileutils import *
from preprocess import *


def create_ondemand_generators(**kwargs):
    n_channels = int(kwargs.get('n_channels', 1))
    out = kwargs.get('output_is_timeseries', False)
    leadtime_conditioning = kwargs.get('leadtime_conditioning', 0)

    start_date = kwargs.get('start_date')
    stop_date = kwargs.get('stop_date')

    filenames = read_filenames(start_date, stop_date)
    assert(len(filenames) > 0)

    datasets = []
    if out:
        i = 0

        while i < (len(filenames) - (n_channels + 1)):
            ds_files = []
            for j in range(n_channels + 1):
                ds_files.append(filenames[i + j])
            datasets.append(ds_files)
            i += (n_channels + 1)
    elif leadtime_conditioning:
        i = 0

        while i < len(filenames) - (n_channels + leadtime_conditioning):
            hist = []
            for j in range(n_channels):
                hist.append(filenames[i+j])


            for j in range(0, leadtime_conditioning):
                hist_ = hist.copy()
                hist_.append('leadtime={}'.format((1+j)*15))
                hist_.append(filenames[i+n_channels+j])

                datasets.append(hist_)
            i += n_channels + leadtime_conditioning

    else:
        i = 0
        while i < len(filenames) - n_channels:
            datasets.append([])
            for j in range(n_channels):
                datasets[-1].append(filenames[i+j]) # training (x) files
            datasets[-1].append(filenames[i+n_channels]) # test (y) file
            i += n_channels + 1

    np.random.shuffle(datasets)
    test_val_split = (np.floor(len(datasets) * 0.9)).astype(np.int)
    train = OnDemandEffectiveCloudinessGenerator(datasets[0:test_val_split], **kwargs)
    val = OnDemandEffectiveCloudinessGenerator(datasets[test_val_split:-1], **kwargs)

    return train, val


def create_generators(**kwargs):

    dataseries_file = kwargs.get('dataseries_file', '')

    if dataseries_file:
        return create_generators_from_dataseries(**kwargs)

    return create_ondemand_generators(**kwargs)


class OnDemandEffectiveCloudinessGenerator(keras.utils.Sequence):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.n_channels = int(kwargs.get('n_channels', 1))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.preprocess = kwargs.get('preprocess', '')
        self.initial = True
        self.include_datetime = kwargs.get('include_datetime', False)
        self.include_environment_data = kwargs.get('include_environment_data', False)
        self.leadtime_conditioning = kwargs.get('leadtime_conditioning', 0)
        self.output_is_timeseries = kwargs.get('output_is_timeseries', False)

        assert(self.n_channels > 0)


    def __len__(self):
        return (np.floor(len(self.dataset) / self.batch_size)).astype(np.int)

    def __getitem__(self, i):
        if self.output_is_timeseries:
            return self.create_convlstm_input(i)
        elif self.leadtime_conditioning:
            return self.create_unet_input_with_leadtime(i)
        else:
            return self.create_unet_input(i)

    def create_convlstm_input(self, i):
        ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for d in ds:
            x.append(preprocess_many(read_gribs(d[0:self.n_channels]), self.preprocess))
            y.append(preprocess_many(read_gribs(d[1:self.n_channels+1]), self.preprocess))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y


    def create_unet_input(self, i):
        batch_ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for ds in batch_ds:
            x.append(preprocess_many(read_gribs(ds[:-1]), self.preprocess))
            y.append(preprocess_single(read_grib(ds[-1]), self.preprocess))

            dt = datetime.datetime.strptime(os.path.basename(ds[-2]).split('_')[0], '%Y%m%dT%H%M%S')

            x[-1] = np.moveaxis(x[-1], 0, 3)
            x[-1] = np.squeeze(x[-1], axis=-2)

            if self.include_datetime:
                dts  = create_datetime(dt, get_img_size(self.preprocess))
                x[-1] = np.concatenate((x[-1], dts[0], dts[1]), axis=-1)

            if self.include_environment_data:
                envs = create_environment_data(self.preprocess)
                x[-1] = np.concatenate((x[-1], envs[0], envs[1]), axis=-1)

        x = np.asarray(x)
        y = np.asarray(y)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y


    def create_unet_input_with_leadtime(self, i):
        batch_ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for ds_ in batch_ds:
            ds = ds_.copy()
            leadtime = int(int(ds[-2].split('=')[1])/15)
            del ds[-2]
            x_ = preprocess_many(read_gribs(ds[:-1]), self.preprocess)
            lt_ = create_squeezed_leadtime_conditioning(get_img_size(self.preprocess), self.leadtime_conditioning, leadtime)
            x.append(np.concatenate((x_, lt_), axis=0))

            dt = datetime.datetime.strptime(os.path.basename(ds[-2]).split('_')[0], '%Y%m%dT%H%M%S')

            x[-1] = np.moveaxis(x[-1], 0, 3)
            x[-1] = np.squeeze(x[-1], axis=-2)

            if self.include_datetime:
                dts  = create_datetime(dt, get_img_size(self.preprocess))
                x[-1] = np.concatenate((x[-1], dts[0], dts[1]), axis=-1)

            if self.include_environment_data:
                envs = create_environment_data(self.preprocess)
                x[-1] = np.concatenate((x[-1], envs[0], envs[1]), axis=-1)

            y.append(preprocess_single(read_grib(ds[-1]), self.preprocess))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y
