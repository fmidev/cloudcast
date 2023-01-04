import datetime
import numpy as np
from tensorflow import keras
from base.fileutils import *
from base.preprocess import *


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
