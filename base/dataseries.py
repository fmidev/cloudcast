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
)
from base.gributils import read_gribs
from base.fileutils import read_filenames


def read_times_from_preformatted_files(dirname):
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


def read_datas_from_preformatted_files(dirname, toc, times):
    datas = []

    for t in times:
        e = toc[t]
        idx = e["index"]
        filename = e["filename"]
        datafile = np.load(filename, mmap_mode="r")
        datas.append(datafile[idx])
    return datas, times


class LazyDataSeries:
    def __init__(self, **kwargs):
        self.batch_size = int(kwargs.get("batch_size"))
        self.n_channels = int(kwargs.get("n_channels"))
        self.img_size = kwargs.get("img_size")
        self.leadtime_conditioning = int(kwargs.get("leadtime_conditioning"))
        self.terrain_type_data = None
        self.topography_data = None
        self.include_datetime = kwargs.get("include_datetime", False)
        self.dataseries_directory = kwargs.get("dataseries_directory", None)

        assert self.leadtime_conditioning > 0  # temporary

        if self.dataseries_directory is not None:
            self.elements, self.toc = read_times_from_preformatted_files(
                self.dataseries_directory
            )

        else:
            start_date = kwargs.get("start_date")
            stop_date = kwargs.get("stop_date")

            self.elements = read_filenames(start_date, stop_date)

        self.initialize(kwargs)

    def __len__(self):
        return self.leadtime_conditioning * int(
            len(self.elements) / (self.n_channels + self.leadtime_conditioning)
        )

    def initialize(self, kwargs):

        if int(kwargs.get("leadtime_conditioning", 0)) > 0:
            leadtimes = np.asarray(
                [
                    create_squeezed_leadtime_conditioning(
                        self.img_size, self.leadtime_conditioning, x
                    )
                    for x in range(self.leadtime_conditioning)
                ]
            )
            self.leadtimes = np.squeeze(leadtimes, 1)

        if kwargs.get("include_topography", False):
            self.topography_data = np.expand_dims(
                create_topography_data(self.img_size), axis=0
            )

        if kwargs.get("include_terrain_type", False):
            self.terrain_type_data = np.expand_dims(
                create_terrain_type_data(self.img_size), axis=0
            )

        n_source = int(
            len(self.elements) / (self.n_channels + self.leadtime_conditioning)
        )
        self._indexes = np.asarray(
            [
                x * (self.n_channels + self.leadtime_conditioning)
                for x in range(n_source)
            ]
        )
        np.random.shuffle(self._indexes)

    def get_dataset(self, take=None, skip=None):
        def gen(indexes):

            for hist_start in indexes:
                x_elems = self.elements[hist_start : hist_start + self.n_channels]
                y_elems = self.elements[
                    hist_start
                    + self.n_channels : hist_start
                    + self.n_channels
                    + self.leadtime_conditioning
                ]

                if self.dataseries_directory is None:
                    x = read_gribs(x_elems, dtype=np.single, disable_preprocess=True)
                    y = read_gribs(y_elems, dtype=np.single, disable_preprocess=True)

                    x = tf.image.resize(x, self.img_size)
                    y = tf.image.resize(y, self.img_size)

                    ts = datetime.strptime(
                        x_elems[-1].split("/")[-1].split("_")[0], "%Y%m%dT%H%M%S"
                    )
                else:
                    x, times = read_datas_from_preformatted_files(
                        self.dataseries_directory, self.toc, x_elems
                    )
                    y, _ = read_datas_from_preformatted_files(
                        self.dataseries_directory, self.toc, y_elems
                    )

                    ts = times[-1]

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
                    x_ = np.squeeze(np.swapaxes(x_, 0, 3))
                    y_ = y[i]

                    yield (x_, y_)

        def resize(x, y, img_size):
            # resize all dimension to correct shape, if they are not already
            if x.shape[1] != img_size[0] or x.shape[2] != img_size[1]:
                x = tf.image.resize(x, img_size)
                y = tf.image.resize(y, img_size)

            return (x, y)

        def flip(x, y, n):
            # flip first n dimensions as they contain the payload data
            # concatenate the flipped data with the rest
            x = tf.concat([tf.image.flip_up_down(x[..., 0:n]), x[..., n:]], axis=-1)
            y = tf.image.flip_up_down(y)
            return (x, y)

        def normalize(x, y, n):
            # normalize input data (x) to 0..1
            # scale output data to 0..1
            # mean, variance = tf.nn.moments(x, axes=[0, 1], keepdims=True)
            # x = (x - mean) / tf.sqrt(variance + tf.keras.backend.epsilon())

            # scale all data to 0..1, to preserve compatibility with older models
            # trained with this software
            x = tf.concat([0.01 * x[..., 0:n], x[..., n:]], axis=-1)
            y = y * 0.01
            return (x, y)

        indexes = None

        if take is not None:
            indexes = self._indexes[0:take]

        if skip is not None:
            indexes = self._indexes[skip:]

        if indexes is None:
            indexes = np.copy(self._indexes)

        x_dim_len = self.n_channels
        x_dim_len += 1 if self.leadtime_conditioning > 0 else 0
        x_dim_len += 2 if self.include_datetime else 0
        x_dim_len += 1 if self.topography_data is not None else 0
        x_dim_len += 1 if self.terrain_type_data is not None else 0

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(
                    shape=self.img_size + (x_dim_len,), dtype=tf.float32, name="x"
                ),
                tf.TensorSpec(shape=self.img_size + (1,), dtype=tf.float32, name="y"),
            ),
            args=(indexes,),
        )

        if self.dataseries_directory is None:
            dataset = (
                dataset.map(lambda x, y: resize(x, y, self.img_size))
                .map(lambda x, y: flip(x, y, self.n_channels))
                .map(lambda x, y: normalize(x, y, self.n_channels))
            )

        dataset = dataset.batch(self.batch_size).prefetch(AUTOTUNE)
        return dataset
