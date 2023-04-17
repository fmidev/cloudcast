import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np
import glob
from datetime import datetime, timedelta
from enum import Enum
import copy
from base.preprocess import (
    create_topography_data,
    create_terrain_type_data,
    create_squeezed_leadtime_conditioning,
    create_datetime,
    create_sun_elevation_angle,
    create_sun_elevation_angle_data,
    get_img_size,
)
from base.gributils import read_gribs
from base.fileutils import read_filenames

OpMode = Enum("OperatingMode", ["TRAIN", "INFER", "VERIFY"])


def read_times_from_preformatted_files_directory(dirname):
    toc = {}
    for f in glob.glob("{}/*-times.npy".format(dirname)):
        times = np.load(f)

        for i, t in enumerate(times):
            toc[t] = {"filename": f.replace("-times", ""), "index": i, "time": t}

    times = list(toc.keys())

    times.sort()
    print("Read {} times from {}".format(len(times), dirname))
    return times, toc


def read_datas_from_preformatted_files_directory(dirname, toc, times):
    datas = []

    for t in times:
        e = toc[t]
        idx = e["index"]
        filename = e["filename"]
        datafile = np.load(filename, mmap_mode="r")
        datas.append(datafile[idx])

    return datas, times


def read_times_from_preformatted_file(filename):
    ds = np.load(filename)
    data = ds["arr_0"]
    times = ds["arr_1"]

    toc = {}
    for i, t in enumerate(times):
        toc[t] = {"index": i, "time": t}

    return times, data, toc


def read_datas_from_preformatted_file(all_times, all_data, req_times, toc):
    datas = []
    for t in req_times:
        index = toc[t]["index"]
        datas.append(all_data[index])

    return datas, req_times


class DataSeriesGenerator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        print(
            "Generator number of batches: {} batch size: {}".format(
                len(self), self.batch_size
            )
        )

    def __len__(self):
        """Return number of batches in this dataset"""
        return len(self.placeholder) // self.batch_size

    def __getitem__(self, idx):
        # placeholder X elements:
        # 0.. n_channels: history of actual data (YYYYMMDDTHHMMSS, string)
        # n_channels    : leadtime conditioning (0..11, int)
        # n_channels + 1: include datetime (bool)
        # n_channels + 2: include topography (bool)
        # n_channels + 3: include terrain type (bool)
        # n_channels + 4: include sun elevation angle (bool)

        ph = self.placeholder[idx]

        X = ph[0]
        Y = ph[1]

        x_hist = X[0 : self.n_channels]

        x, y, xtimes, ytimes = self.get_xy(x_hist, [Y])

        lc = X[self.n_channels]
        lt = np.expand_dims(self.leadtimes[lc], axis=0)

        x = np.asarray(x)
        y = np.asarray(y)

        y = np.squeeze(y, axis=0)

        x = np.concatenate((x, lt), axis=0)

        ts = datetime.strptime(xtimes[-1], "%Y%m%dT%H%M%S")  # "analysis time"
        y_time = ts + timedelta(minutes=(1 + lc) * 15)

        if X[self.n_channels + 1]:
            tod, toy = create_datetime(y_time, self.img_size)
            tod = np.expand_dims(tod, axis=0)
            toy = np.expand_dims(toy, axis=0)
            x = np.concatenate((x, tod, toy), axis=0)

        if X[self.n_channels + 2]:
            x = np.concatenate((x, self.topography_data), axis=0)

        if X[self.n_channels + 3]:
            x = np.concatenate((x, self.terrain_type_data), axis=0)

        if X[self.n_channels + 4]:
            angle = self.sun_elevation_angle_data[
                y_time.replace(year=2023).strftime("%Y%m%dT%H%M%S")
            ]
            angle = np.expand_dims(angle, axis=0)
            angle = tf.image.resize(angle, self.img_size)
            x = np.concatenate((x, angle), axis=0)

        x = np.squeeze(np.swapaxes(x, 0, 3))

        if self.operating_mode in (OpMode.VERIFY, OpMode.INFER):
            return (
                x,
                y,
                np.append(xtimes, y_time.strftime("%Y%m%dT%H%M%S")),
            )
        else:
            return (x, y)

    def get_xy(self, x_elems, y_elems):
        xtimes = []
        ytimes = []

        if self.dataseries_file is not None:
            x, xtimes = read_datas_from_preformatted_file(
                self.elements, self.data, x_elems, self.toc
            )
            y, ytimes = read_datas_from_preformatted_file(
                self.elements, self.data, y_elems, self.toc
            )

        elif self.dataseries_directory is not None:
            x, xtimes = read_datas_from_preformatted_files_directory(
                self.dataseries_directory, self.toc, x_elems
            )
            y, ytimes = read_datas_from_preformatted_files_directory(
                self.dataseries_directory, self.toc, y_elems
            )

        else:
            x = read_gribs(
                x_elems,
                dtype=np.single,
                disable_preprocess=True,
                enable_cache=self.cache,
                print_filename=self.debug,
            )

            x = tf.image.resize(x, self.img_size)

            xtimes = list(map(lambda x: x.split("/")[-1].split("_")[0], x_elems))

            if self.operating_mode in (OpMode.TRAIN, OpMode.VERIFY):
                y = read_gribs(
                    y_elems,
                    dtype=np.single,
                    disable_preprocess=True,
                    enable_cache=self.cache,
                    print_filename=self.debug,
                )

                y = tf.image.resize(y, self.img_size)

                ytimes = list(map(lambda x: x.split("/")[-1].split("_")[0], y_elems))
            else:
                y = np.full((1,) + self.img_size + (1,), np.NaN)

        return x, y, xtimes, ytimes

    def __call__(self):
        for i in range(len(self.placeholder)):
            elem = self.__getitem__(i)
            yield elem

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_data:
            np.random.shuffle(self.placeholder)


class LazyDataSeries:
    def __init__(self, **kwargs):
        try:
            opts = kwargs["opts"]
            self.n_channels = opts.n_channels
            self.leadtime_conditioning = int(
                kwargs.get("leadtime_conditioning", opts.leadtime_conditioning)
            )
            self.img_size = get_img_size(opts.preprocess)
            self.include_datetime = opts.include_datetime
            self.include_topography = opts.include_topography
            self.include_terrain_type = opts.include_terrain_type
            self.include_sun_elevation_angle = opts.include_sun_elevation_angle

        except KeyError:
            self.n_channels = int(kwargs.get("n_channels"))
            self.img_size = kwargs.get("img_size")
            self.leadtime_conditioning = int(kwargs.get("leadtime_conditioning"))
            self.include_datetime = kwargs.get("include_datetime", False)
            self.include_topography = kwargs.get("include_topography", False)
            self.include_terrain_type = kwargs.get("include_terrain_type", False)
            self.include_sun_elevation_angle = kwargs.get(
                "include_sun_elevation_angle", False
            )

        self.batch_size = int(kwargs.get("batch_size", 1))
        self.dataseries_file = kwargs.get("dataseries_file", None)
        self.dataseries_directory = kwargs.get("dataseries_directory", None)
        self.start_date = kwargs.get("start_date", None)
        self.stop_date = kwargs.get("stop_date", None)
        self.reuse_y_as_x = kwargs.get("reuse_y_as_x", False)
        self.shuffle_data = kwargs.get("shuffle_data", True)
        self.debug = kwargs.get("enable_debug", False)
        self.filenames = kwargs.get("filenames", None)
        self.analysis_time = kwargs.get("analysis_time", None)
        operating_mode = kwargs.get("operating_mode", "TRAIN")

        self.cache = kwargs.get("enable_cache", False)

        if operating_mode == "TRAIN":
            self.operating_mode = OpMode.TRAIN
        elif operating_mode == "INFER":
            self.operating_mode = OpMode.INFER
        elif operating_mode == "VERIFY":
            self.operating_mode = OpMode.VERIFY
        else:
            print("Invalid operating mode: {}".format(operating_mode))
            sys.exit(1)

        if self.operating_mode == OpMode.INFER:
            self.shuffle_data = False
            self.batch_size = 1

        elif self.operating_mode == OpMode.VERIFY:
            self.shuffle_data = False
            self.batch_size = 1

        self._placeholder = []

        # reuse_y_as_x is True:
        # first set   second set
        # AB CDEF     BC DEFG
        # -->         -->
        # AB C        BC D
        # AB D        BC E
        # AB E        BC F
        # AB F        BD G

        # reuse_y_as_x is False:
        # first set   second set
        # AB CDEF     GH IJKL
        # -->         -->
        # AB C        GH I
        # AB D        GH J
        # AB E        GH K
        # AB F        GH L

        self.initialize()

    def initialize(self):
        # Read static datas, so that each dataset generator
        # does not have to read them

        if self.leadtime_conditioning > 0:
            leadtimes = np.asarray(
                [
                    create_squeezed_leadtime_conditioning(
                        self.img_size, self.leadtime_conditioning, x
                    )
                    for x in range(self.leadtime_conditioning)
                ]
            )
            self.leadtimes = np.squeeze(leadtimes, 1)

        if self.include_topography:
            self.topography_data = np.expand_dims(
                create_topography_data(self.img_size), axis=0
            )

        if self.include_terrain_type:
            self.terrain_type_data = np.expand_dims(
                create_terrain_type_data(self.img_size), axis=0
            )

        if self.include_sun_elevation_angle:
            if self.operating_mode == OpMode.INFER:
                self.sun_elevation_angle_data = {}
                for i in range(self.leadtime_conditioning):
                    ts = self.analysis_time + timedelta(minutes=(1 + i) * 15)
                    self.sun_elevation_angle_data[
                        ts.strftime("%Y%m%dT%H%M%S")
                    ] = create_sun_elevation_angle(ts, (128, 128))
            else:
                self.sun_elevation_angle_data = create_sun_elevation_angle_data(
                    self.img_size,
                )

        # create placeholder data

        if self.dataseries_file is not None:
            self.elements, self.data, self.toc = read_times_from_preformatted_file(
                self.dataseries_file
            )

        elif self.dataseries_directory is not None:
            self.elements, self.toc = read_times_from_preformatted_files_directory(
                self.dataseries_directory
            )

        else:
            if self.filenames is not None:
                self.elements = self.filenames
            else:
                self.elements = read_filenames(self.start_date, self.stop_date)
            self.elements.sort()

        i = 0

        step = 1 if self.reuse_y_as_x else self.n_channels + self.leadtime_conditioning

        n_fut = self.leadtime_conditioning if self.operating_mode != OpMode.INFER else 0

        assert (
            len(self.elements) - (self.n_channels + n_fut)
        ) >= 0, "Too few data to make a prediction: {} (need at least {})".format(
            len(self.elements), self.n_channels + n_fut
        )

        while i <= len(self.elements) - (self.n_channels + n_fut):
            x = list(self.elements[i : i + self.n_channels])

            for lt in range(self.leadtime_conditioning):
                x_ = copy.deepcopy(x)
                x_.append(lt)
                x_.append(self.include_datetime)
                x_.append(self.include_topography)
                x_.append(self.include_terrain_type)
                x_.append(self.include_sun_elevation_angle)

                if self.operating_mode == OpMode.INFER:
                    y = "nan"  # datetime.strptime(self.elements[-1], '%Y%m%dT%H%M%S') + timedelta(minutes=i*15)
                else:
                    y = self.elements[i + self.n_channels + lt]

                self._placeholder.append([x_, y])

            i += step

        assert len(self._placeholder) > 0, "Placeholder array is empty"

        print(
            "Placeholder timeseries length: {} number of samples: {}".format(
                len(self.elements), len(self._placeholder)
            )
        )

        if self.shuffle_data:
            np.random.shuffle(self._placeholder)

    def __len__(self):
        """Return number of samples"""
        return len(self._placeholder)

    def get_dataset(self, take_ratio=None, skip_ratio=None):
        def flip(x, y, t, n):
            # flip first n dimensions as they contain the payload data
            # concatenate the flipped data with the rest
            x = tf.concat([tf.image.flip_up_down(x[..., 0:n]), x[..., n:]], axis=-1)
            y = tf.image.flip_up_down(y)
            return (x, y, t)

        def normalize(x, y, t, n):
            # normalize input data (x) to 0..1
            # scale output data to 0..1
            # mean, variance = tf.nn.moments(x, axes=[0, 1], keepdims=True)
            # x = (x - mean) / tf.sqrt(variance + tf.keras.backend.epsilon())

            if tf.math.reduce_max(x[..., 0]) <= 1.01:
                return (x, y, t)

            # scale all data to 0..1, to preserve compatibility with older models
            # trained with this software
            x = tf.concat([0.01 * x[..., 0:n], x[..., n:]], axis=-1)
            y = y * 0.01
            return (x, y, t)

        placeholder = None

        if take_ratio is not None:
            l = int(len(self._placeholder) * take_ratio)
            placeholder = self._placeholder[0:l]

        if skip_ratio is not None:
            l = int(len(self._placeholder) * skip_ratio)
            placeholder = self._placeholder[l:]

        if placeholder is None:
            placeholder = copy.deepcopy(self._placeholder)

        x_dim_len = self.n_channels
        x_dim_len += 1 if self.leadtime_conditioning > 0 else 0
        x_dim_len += 2 if self.include_datetime else 0
        x_dim_len += 1 if self.include_topography else 0
        x_dim_len += 1 if self.include_terrain_type else 0
        x_dim_len += 1 if self.include_sun_elevation_angle else 0

        sig = (
            tf.TensorSpec(
                shape=self.img_size + (x_dim_len,), dtype=tf.float32, name="x"
            ),
            tf.TensorSpec(shape=self.img_size + (1,), dtype=tf.float32, name="y"),
        )

        if self.operating_mode in (OpMode.INFER, OpMode.VERIFY):
            sig += (
                tf.TensorSpec(
                    shape=(self.n_channels + 1,), dtype=tf.string, name="times"
                ),
            )

        gen = DataSeriesGenerator(placeholder=placeholder, **self.__dict__)
        dataset = tf.data.Dataset.from_generator(gen, output_signature=sig)

        if (
            self.operating_mode != OpMode.TRAIN
            and self.dataseries_directory is None
            and self.dataseries_file is None
        ):
            dataset = dataset.map(lambda x, y, t: flip(x, y, t, self.n_channels)).map(
                lambda x, y, t: normalize(x, y, t, self.n_channels)
            )

        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        return dataset
