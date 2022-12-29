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


class LazyDataSeries:
    def __init__(self, **kwargs):
        self.files = read_filenames(kwargs.get("start_date"), kwargs.get("stop_date"))
        self.batch_size = int(kwargs.get("batch_size"))
        self.n_channels = int(kwargs.get("n_channels"))
        self.img_size = kwargs.get("img_size")
        self.leadtime_conditioning = int(kwargs.get("leadtime_conditioning"))

        self.initialize(kwargs)

    def __len__(self):
        return len(self.files)

    def initialize(self, kwargs):
        x_dim_len = self.n_channels

        if self.leadtime_conditioning > 0:
            x_dim_len += 1

        if kwargs.get("include_datetime", False):
            leadtimes = np.asarray(
                [
                    create_squeezed_leadtime_conditioning(
                        self.img_size, self.leadtime_conditioning, x
                    )
                    for x in range(self.leadtime_conditioning)
                ]
            )
            self.leadtimes = np.squeeze(leadtimes, 1)

            x_dim_len += 2

        if kwargs.get("include_topography", False):
            self.topography_data = np.expand_dims(
                create_topography_data(self.img_size), axis=0
            )
            x_dim_len += 1
        if kwargs.get("include_terrain_type", False):
            self.terrain_type_data = np.expand_dims(
                create_terrain_type_data(self.img_size), axis=0
            )
            x_dim_len += 1

        self.dataset = tf.data.Dataset.from_generator(
            self.gen,
            output_signature=(
                tf.TensorSpec(shape=self.img_size + (x_dim_len,), dtype=tf.float32),
                tf.TensorSpec(shape=self.img_size + (1,), dtype=tf.float32),
            ),
        )

        self.dataset = self.dataset.batch(self.batch_size).cache().prefetch(AUTOTUNE)


    def gen(self):
        hist_start = 0

        assert self.leadtime_conditioning > 0  # temporary
        while hist_start < len(self.files):
            x_files = self.files[hist_start : hist_start + self.n_channels]
            y_files = self.files[
                hist_start
                + self.n_channels : hist_start
                + self.n_channels
                + self.leadtime_conditioning
            ]

            x = read_gribs(x_files, img_size=self.img_size)
            y = read_gribs(y_files, img_size=self.img_size)

            ts = datetime.strptime(
                x_files[-1].split("/")[-1].split("_")[0], "%Y%m%dT%H%M%S"
            )

            tod, toy = create_datetime(ts, self.img_size)
            tod = np.expand_dims(tod, axis=0)
            toy = np.expand_dims(toy, axis=0)

            for i in range(len(y_files)):
                lt = np.expand_dims(self.leadtimes[i], axis=0)
                x_ = np.concatenate((x, lt, tod, toy), axis=0)

                if self.topography_data is not None:
                    x_ = np.concatenate((x_, self.topography_data), axis=0)
                if self.terrain_type_data is not None:
                    x_ = np.concatenate((x_, self.terrain_type_data), axis=0)
                x_ = np.squeeze(np.swapaxes(x_, 0, 3))
                y_ = y[i]

                yield (x_, y_)

            hist_start += self.n_channels + self.leadtime_conditioning
