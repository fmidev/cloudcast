import datetime
import numpy as np
from tensorflow import keras
from fileutils import *

def add_auxiliary_data(x, include_datetime, include_environment_data, dt, preprocess):
    if include_datetime:
        dts = create_datetime(dt, get_img_size(preprocess))
        x = np.concatenate((x, np.expand_dims(dts[0], axis=0), np.expand_dims(dts[1], axis=0)), axis=0)

    if include_environment_data:
        envs = create_environment_data('preprocess')
        x = np.concatenate((x, np.expand_dims(envs[0], axis=0), np.expand_dims(envs[1], axis=0)), axis=0)

    return x

def create_generators_from_dataseries(**kwargs):
    n_channels = int(kwargs.get('n_channels', 1))
    out = kwargs.get('output_is_timeseries', False)
    leadtime_conditioning = kwargs.get('leadtime_conditioning', 0)
    include_datetime = kwargs.get('include_datetime')
    include_environment_data = kwargs.get('include_environment_data')

    dataseries_file = kwargs.get('dataseries_file', '')

    print(f'Reading input data from {dataseries_file}')
    dataset = np.load(dataseries_file)
    dataseries = dataset['arr_0']
    times = dataset['arr_1']

    datasets = []

    print('Creating generators')

    if out:
        i = 0

        n_channels += 1
        while i < dataseries.shape[0] - n_channels:
            ds_data = []
            for j in range(n_channels):
                ds_data.append(dataseries[i + j])
            datasets.append(ds_data)
            i += n_channels

    else:
        i = 0

        n_fut = max(1, leadtime_conditioning)
        while i < dataseries.shape[0] - (n_channels + n_fut):
            hist = np.squeeze(dataseries[i:i+n_channels], axis=-1)
            thist = times[i:i+n_channels]

            assert(len(thist) >= 1)
            dt = datetime.datetime.strptime(thist[-1], '%Y%m%dT%H%M%S') # #12] if n_channels > 1 else thist[0]

            if leadtime_conditioning == 0:
                hist = add_auxiliary_data(hist, include_datetime, include_environment_data, dt, kwargs.get('preprocess'))
                y = np.expand_dims(np.squeeze(dataseries[i+n_channels], axis=-1), axis=0) # y
                datasets.append(np.concatenate((hist, y), axis=0))
                datasets[-1] = np.squeeze(np.swapaxes(datasets[-1], 0, 3))
            else:
                for j in range(0, leadtime_conditioning):
                    leadtime = create_squeezed_leadtime_conditioning(img_size, leadtime_conditioning, j) # x leadtime conditioning

                    x = np.concatenate((hist, leadtime), axis=0)
                    x = add_auxiliary_data(x, include_datetime, include_environment_data, dt, kwargs.get('preprocess'))

                    y = np.expand_dims(dataseries[i+n_channels+j], axis=0) # y

                    datasets.append(np.concatenate((x, y), axis=0))
                    datasets[-1] = np.squeeze(np.swapaxes(datasets[-1], 0, 3))

            i += n_channels + leadtime_conditioning

    np.random.shuffle(datasets)
    test_val_split = (np.floor(len(datasets) * 0.9)).astype(np.int)
    train = EffectiveCloudinessGenerator(datasets[0:test_val_split], **kwargs)
    val = EffectiveCloudinessGenerator(datasets[test_val_split:-1], **kwargs)

    return train, val


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


class EffectiveCloudinessGenerator(keras.utils.Sequence):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.initial = True
        self.output_is_timeseries = kwargs.get('output_is_timeseries', False)


    def __len__(self):
        return (np.floor(len(self.dataset) / self.batch_size)).astype(np.int)

    def __getitem__(self, i):
        if self.output_is_timeseries:
            return self.create_convlstm_input(i)
        else:
            return self.create_unet_input(i)

    def create_convlstm_input(self, i):
        batch_ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for d in batch_ds:
            x.append(d[:-1])
            y.append(d[1:])

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

        for i in batch_ds:
            x.append(i[...,:-1])
            y.append(np.expand_dims(i[...,-1], axis=-1))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.initial:
            print(f'Batch shapes: x {x.shape} y {y.shape}')
            self.initial = False

        return x, y



# datetime ring buffer
#
# datetimes are generated so that forecast analysis time fits between the given range
#
# for example
# * history length = 2
# * prediction length = 2
# * single time = 202111T0800
#
# times are:
# - 20211107T2330
# - 20211107T2345
# - 20211108T0000
# - 20211108T0015


class TimeseriesGenerator:
    def __init__(self, start_date, history_len, pred_len, step=datetime.timedelta(minutes=15), stop_date=None):
        self.date = start_date
        self.stop_date = stop_date
        self.history_len = history_len
        self.prediction_len = pred_len
        self.step = step
        self.times = [start_date - (history_len - 1) * step]
        self.create()

    def __iter__(self):
        while True:
            yield self.times
            self.create()
            if self.times[-1] > self.stop_date:
                break

    def create(self):
        if len(self.times) > 1:
            self.times.pop(0)
        while len(self.times) < self.history_len + self.prediction_len:
            self.times.append(self.times[-1] + self.step)

