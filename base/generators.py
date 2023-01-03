import datetime
import numpy as np
from tensorflow import keras
from base.fileutils import *
from base.preprocess import *


def add_auxiliary_data(
    x,
    include_datetime,
    include_topography_data,
    include_terrain_type_data,
    dt,
    preprocess,
):
    if include_datetime:
        dts = create_datetime(dt, get_img_size(preprocess))
        x = np.concatenate(
            (x, np.expand_dims(dts[0], axis=0), np.expand_dims(dts[1], axis=0)), axis=0
        )

    if include_topography_data:
        topo = create_topography_data(get_img_size(preprocess))
        x = np.concatenate((x, np.expand_dims(topo, axis=0)), axis=0)

    if include_terrain_type_data:
        terr = create_terrain_type_data(get_img_size(preprocess))
        x = np.concatenate((x, np.expand_dims(terr, axis=0)), axis=0)

    return x


def create_generators_from_dataseries(**kwargs):
    n_channels = int(kwargs.get("n_channels", 1))
    out = kwargs.get("output_is_timeseries", False)
    leadtime_conditioning = kwargs.get("leadtime_conditioning", 0)
    include_datetime = kwargs.get("include_datetime")
    include_topography_data = kwargs.get("include_topography")
    include_terrain_type_data = kwargs.get("include_terrain_type")
    preprocess = kwargs.get("preprocess")
    dataseries_file = kwargs.get("dataseries_file", "")
    onehot_encoding = kwargs.get("onehot_encoding", False)

    print(f"Reading input data from {dataseries_file}")
    dataset = np.load(dataseries_file)
    dataseries = dataset["arr_0"]
    times = dataset["arr_1"]

    datasets = []

    print("Creating generators")

    if out:
        i = 0

        if include_topography_data:
            topography_data = create_topography_data(get_img_size(preprocess))
        if include_terrain_type_data:
            terrain_type_data = create_terrain_type_data(get_img_size(preprocess))

        n_channels += 1
        while i < dataseries.shape[0] - n_channels:
            ds_data = []
            for j in range(n_channels):
                x = dataseries[i + j]

                if include_topography_data:
                    x = np.concatenate((x, topography_data), axis=-1)
                if include_terrain_type_data:
                    x = np.concatenate((x, terrain_type_data), axis=-1)

                ds_data.append(x)
            datasets.append(ds_data)
            i += n_channels

    else:
        i = 0

        # number of predictions
        n_fut = max(1, leadtime_conditioning)

        # with n_channels=4, leadtime_conditioning=6:
        # HHHHPPPPPPHHHHPPPPPP...

        # if leadtime conditioning is set, leadtime will be a part
        # of the input tensor

        while i < dataseries.shape[0] - (n_channels + n_fut):
            # read history data
            hist = dataseries[i : i + n_channels]

            # Time of the last history data, we use that to create a time tensor
            # if that's needed (requested)
            dt = datetime.datetime.strptime(times[i + n_channels], "%Y%m%dT%H%M%S")

            if leadtime_conditioning == 0:
                x = add_auxiliary_data(
                    hist,
                    include_datetime,
                    include_topography_data,
                    include_terrain_type_data,
                    dt,
                    preprocess,
                )
                y = np.expand_dims(
                    np.expand_dims(
                        np.squeeze(dataseries[i + n_channels], axis=-1), axis=0
                    ),
                    axis=-1,
                )
                datasets.append(np.concatenate((x, y), axis=0))
                datasets[-1] = np.squeeze(np.swapaxes(datasets[-1], 0, 3))
            else:

                for j in range(n_fut):
                    if not onehot_encoding:
                        leadtime = create_squeezed_leadtime_conditioning(
                            get_img_size(preprocess), leadtime_conditioning, j
                        )
                    else:
                        leadtime = create_onehot_leadtime_conditioning(
                            get_img_size(preprocess), leadtime_conditioning, j
                        )
                    x = np.concatenate((hist, leadtime), axis=0)
                    x = add_auxiliary_data(
                        x,
                        include_datetime,
                        include_topography_data,
                        include_terrain_type_data,
                        dt,
                        preprocess,
                    )
                    y = np.expand_dims(
                        np.expand_dims(
                            np.squeeze(dataseries[i + n_channels + j], axis=-1), axis=0
                        ),
                        axis=-1,
                    )

                    datasets.append(np.concatenate((x, y), axis=0))
                    datasets[-1] = np.squeeze(np.swapaxes(datasets[-1], 0, 3))

            i += n_channels + leadtime_conditioning

        assert len(datasets) == int(dataseries.shape[0] / (n_channels + n_fut)) * n_fut

    np.random.shuffle(datasets)
    test_val_split = (np.floor(len(datasets) * 0.9)).astype(np.int)
    train = EffectiveCloudinessGenerator(datasets[0:test_val_split], **kwargs)
    val = EffectiveCloudinessGenerator(datasets[test_val_split:-1], **kwargs)

    return train, val


def create_generators(**kwargs):

    dataseries_file = kwargs.get("dataseries_file", "")

    if dataseries_file:
        return create_generators_from_dataseries(**kwargs)

    return create_ondemand_generators(**kwargs)


class EffectiveCloudinessGenerator(keras.utils.Sequence):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.initial = True
        self.output_is_timeseries = kwargs.get("output_is_timeseries", False)

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
            print(f"Batch shapes: x {x.shape} y {y.shape}")
            self.initial = False

        return x, y

    def create_unet_input(self, i):
        batch_ds = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

        x = []
        y = []

        for i in batch_ds:
            x.append(i[..., :-1])
            y.append(np.expand_dims(i[..., -1], axis=-1))

        x = np.asarray(x)
        y = np.asarray(y)

        if self.initial:
            print(f"Batch shapes: x {x.shape} y {y.shape}")
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
    def __init__(
        self,
        start_date,
        stop_date,
        history_len,
        pred_len,
        step=datetime.timedelta(minutes=15),
    ):
        self.date = start_date
        self.stop_date = stop_date
        self.history_len = history_len
        self.prediction_len = pred_len
        self.step = step
        self.times = [start_date - (history_len - 1) * step]
        self.create()
        assert start_date is not None and stop_date is not None

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


class DataSeries:
    def __init__(
        self,
        producer,
        preprocess=None,
        single_analysis_time=True,
        param="effective-cloudiness",
        fill_gaps_max=0,
        cache_data=True,
    ):
        self.data_series = {}
        self.producer = producer
        self.preprocess = preprocess
        self.analysis_time = None
        self.single_analysis_time = single_analysis_time
        self.param = param
        self.fill_gaps_max = fill_gaps_max
        self.cache_data = cache_data

    def fill_gaps(self, series, analysis_time):
        new_series = {}
        gaps_filled = 0

        for i, s in enumerate(series):
            ismiss = np.isnan(series[s]).any()

            if not ismiss:
                new_series[s] = series[s]
                continue

            if gaps_filled == self.fill_gaps_max:
                print("Maximum gaps filled reached ({})".format(self.fill_gaps_max))
                new_series[s] = series[s]
                continue

            new_first_time = None
            if i == 0:
                new_first_time = s - datetime.timedelta(minutes=15)
            elif i == len(series) - 1:
                new_first_time = list(new_series.keys())[0] - datetime.timedelta(
                    minutes=15
                )

            if new_first_time is not None:
                print("Gap-filling for {}".format(s))
                new_first_data = preprocess_single(
                    read_time(
                        new_first_time,
                        self.producer,
                        analysis_time,
                        print_filename=True,
                        param=self.param,
                    ),
                    self.preprocess,
                )
                new_series[new_first_time] = new_first_data

            if i > 0 and i < len(series) - 1:
                prev_time = list(new_series.keys())[i - 1]
                next_time = list(series.keys())[i + 1]
                prev_data = new_series[prev_time]
                next_data = series[next_time]
                # linear interpolation between two points
                new_data = np.asarray(
                    [
                        np.interp(0.5, [0, 1], [x, y])
                        for x, y in zip(prev_data.ravel(), next_data.ravel())
                    ]
                ).reshape(prev_data.shape)
                print(
                    "Interpolating for {} (between {} and {})".format(
                        s, prev_time, next_time
                    )
                )
                new_series[s] = new_data

            gaps_filled += 1
        sorted_series = {}
        for s in sorted(new_series):
            sorted_series[s] = new_series[s]
        new_series = sorted_series

        assert len(new_series) == len(series)
        return new_series

    def read_data(self, times, analysis_time=None):
        datakeys = list(self.data_series.keys())

        if analysis_time != self.analysis_time and self.single_analysis_time:
            datakeys = []

        new_series = {}

        if self.cache_data:
            new_times = list(set(datakeys + times))
        else:
            new_times = times

        new_times.sort()

        for t in new_times:
            if t in datakeys:
                new_series[t] = self.data_series[t]
            else:
                new_series[t] = preprocess_single(
                    read_time(
                        t,
                        self.producer,
                        analysis_time,
                        print_filename=True,
                        param=self.param,
                    ),
                    self.preprocess,
                )

        if self.fill_gaps_max > 0:
            new_series = self.fill_gaps(new_series, analysis_time)

        self.data_series = new_series
        self.analysis_time = analysis_time

        return np.asarray(list(self.data_series.values()))
