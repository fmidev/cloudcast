import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np
import glob
from datetime import datetime
from base.preprocess import (
    create_topography_data,
    create_terrain_type_data,
    create_squeezed_leadtime_conditioning,
    create_datetime,
    create_sun_elevation_angle,
    get_img_size,
)
from base.gributils import read_gribs
from base.fileutils import read_filenames


def read_times_from_preformatted_files_directory(dirname):
    toc = {}
    for f in glob.glob("{}/*-times.npy".format(dirname)):
        arr = np.load(f)
        times = list(map(lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"), arr))

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
    times = list(map(lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"), times))

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


class LazyDataSeries:
    def __init__(self, **kwargs):
        try:
            opts = kwargs["opts"]
            self.n_channels = opts.n_channels
            self.leadtime_conditioning = opts.leadtime_conditioning
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
        self.terrain_type_data = None
        self.topography_data = None

        self.dataseries_file = kwargs.get("dataseries_file", None)
        self.dataseries_directory = kwargs.get("dataseries_directory", None)
        self.start_date = kwargs.get("start_date", None)
        self.stop_date = kwargs.get("stop_date", None)
        self.reuse_y_as_x = kwargs.get("reuse_y_as_x", False)
        self.shuffle_data = kwargs.get("shuffle_data", True)
        self.debug = kwargs.get("enable_debug", False)

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

    def __len__(self):
        """Return number of samples"""

        if self.reuse_y_as_x:
            return len(self._indexes) * self.leadtime_conditioning

        return self.leadtime_conditioning * int(
            len(self.elements) / (self.n_channels + self.leadtime_conditioning)
        )

    def initialize(self):

        assert self.leadtime_conditioning > 0  # temporary

        if self.dataseries_file is not None:
            self.elements, self.data, self.toc = read_times_from_preformatted_file(
                self.dataseries_file
            )

        elif self.dataseries_directory is not None:
            self.elements, self.toc = read_times_from_preformatted_files_directory(
                self.dataseries_directory
            )

        else:
            self.elements = read_filenames(self.start_date, self.stop_date)
            self.elements.sort()

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

        if self.reuse_y_as_x:
            self._indexes = np.arange(
                0, len(self.elements) - self.n_channels - self.leadtime_conditioning
            )

        else:
            n_source = int(
                len(self.elements) / (self.n_channels + self.leadtime_conditioning)
            )
            self._indexes = np.asarray(
                [
                    x * (self.n_channels + self.leadtime_conditioning)
                    for x in range(n_source)
                ]
            )

        if self.shuffle_data:
            np.random.shuffle(self._indexes)

    def get_dataset(self, take_ratio=None, skip_ratio=None):
        def gen(indexes):

            for hist_start in indexes:
                x_elems = self.elements[hist_start : hist_start + self.n_channels]
                y_elems = self.elements[
                    hist_start
                    + self.n_channels : hist_start
                    + self.n_channels
                    + self.leadtime_conditioning
                ]

                xtimes = []
                ytimes = []

                if self.dataseries_file is not None:
                    x, xtimes = read_datas_from_preformatted_file(
                        self.elements, self.data, x_elems, self.toc
                    )
                    y, ytimes = read_datas_from_preformatted_file(
                        self.elements, self.data, y_elems, self.toc
                    )

                    ts = xtimes[-1]

                elif self.dataseries_directory is not None:
                    x, xtimes = read_datas_from_preformatted_files_directory(
                        self.dataseries_directory, self.toc, x_elems
                    )
                    y, ytimes = read_datas_from_preformatted_files_directory(
                        self.dataseries_directory, self.toc, y_elems
                    )

                    ts = xtimes[-1]

                else:
                    x = read_gribs(
                        x_elems,
                        dtype=np.single,
                        disable_preprocess=True,
                        print_filename=self.debug,
                    )
                    y = read_gribs(
                        y_elems,
                        dtype=np.single,
                        disable_preprocess=True,
                        print_filename=self.debug,
                    )

                    x = tf.image.resize(x, self.img_size)
                    y = tf.image.resize(y, self.img_size)

                    xtimes = list(
                        map(lambda x: x.split("/")[-1].split("_")[0], x_elems)
                    )
                    ytimes = list(
                        map(lambda x: x.split("/")[-1].split("_")[0], y_elems)
                    )

                    ts = datetime.strptime(xtimes[-1], "%Y%m%dT%H%M%S")

                if self.include_datetime:
                    tod, toy = create_datetime(ts, self.img_size)
                    tod = np.expand_dims(tod, axis=0)
                    toy = np.expand_dims(toy, axis=0)

                for i in range(len(y)):
                    lt = np.expand_dims(self.leadtimes[i], axis=0)
                    x_ = np.concatenate((x, lt), axis=0)

                    if self.include_datetime:
                        x_ = np.concatenate((x_, tod, toy), axis=0)
                    if self.topography_data is not None:
                        x_ = np.concatenate((x_, self.topography_data), axis=0)
                    if self.terrain_type_data is not None:
                        x_ = np.concatenate((x_, self.terrain_type_data), axis=0)
                    if self.include_sun_elevation_angle:
                        angle = create_sun_elevation_angle(ts, self.img_size)
                        angle = np.expand_dims(angle, axis=0)
                        x_ = np.concatenate((x_, angle), axis=0)

                    x_ = np.squeeze(np.swapaxes(x_, 0, 3))
                    y_ = y[i]

                    yield (
                        x_,
                        y_,
                        np.asarray(xtimes + [ytimes[i]]),
                    )

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

            # scale all data to 0..1, to preserve compatibility with older models
            # trained with this software
            x = tf.concat([0.01 * x[..., 0:n], x[..., n:]], axis=-1)
            y = y * 0.01
            return (x, y, t)

        indexes = None

        if take_ratio is not None:
            l = int(len(self._indexes) * take_ratio)
            indexes = self._indexes[0:l]

        if skip_ratio is not None:
            l = int(len(self._indexes) * skip_ratio)
            indexes = self._indexes[l:]

        if indexes is None:
            indexes = np.copy(self._indexes)

        x_dim_len = self.n_channels
        x_dim_len += 1 if self.leadtime_conditioning > 0 else 0
        x_dim_len += 2 if self.include_datetime else 0
        x_dim_len += 1 if self.topography_data is not None else 0
        x_dim_len += 1 if self.terrain_type_data is not None else 0
        x_dim_len += 1 if self.include_sun_elevation_angle else 0

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(
                    shape=self.img_size + (x_dim_len,), dtype=tf.float32, name="x"
                ),
                tf.TensorSpec(shape=self.img_size + (1,), dtype=tf.float32, name="y"),
                tf.TensorSpec(
                    shape=(self.n_channels + 1,), dtype=tf.string, name="times"
                ),
            ),
            args=(indexes,),
        )

        if self.dataseries_directory is None:
            dataset = dataset.map(lambda x, y, t: flip(x, y, t, self.n_channels)).map(
                lambda x, y, t: normalize(x, y, t, self.n_channels)
            )

        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        return dataset
