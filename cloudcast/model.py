import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    Cropping2D,
    Concatenate,
    ConvLSTM2D,
    BatchNormalization,
    Conv3D,
    Activation,
)
from tensorflow.keras.models import Model

from fss import make_FSS_loss
from ssim import make_SSIM_loss, make_MS_SSIM_loss
from ks import make_KS_loss
from bcl1 import make_bc_l1_loss

from tensorflow.keras import mixed_precision
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == "GPU"]


def get_compute_capability(gpu_id=0):
    devices = tf.config.experimental.list_physical_devices()

    for d in devices:
        if d[1] == "GPU" and int(d[0][-1]) == gpu_id:
            details = tf.config.experimental.get_device_details(d)
            return details["compute_capability"]

    return None


cc = get_compute_capability()

if cc is not None and int(cc[0]) >= 7:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

policy = tf.keras.mixed_precision.global_policy()

print(
    "Compute dtype: {} Variable dtype: {} Number of GPUs: {}".format(
        policy.compute_dtype, policy.variable_dtype, len(get_available_gpus())
    )
)


def get_loss_function(loss_function):
    if loss_function.startswith("ssim"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_SSIM_loss()

        return make_SSIM_loss(int(values[1]))
    elif loss_function.startswith("msssim"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_MS_SSIM_loss()

        return make_SSIM_loss(int(values[1]))
    elif loss_function == "bcl1":
        return make_bc_l1_loss()
    elif loss_function.startswith("fss"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_FSS_loss(5)
        assert len(values) == 3

        mask = int(values[1])
        bins = list(map(lambda x: float(x), values[2].split(",")))

        b = []
        for i in range(len(bins) - 1):
            b.append([bins[i], bins[i + 1]])

        bins = tf.constant(b)
        return make_FSS_loss(mask, bins, hard_discretization=False)
    elif loss_function.startswith("ks"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_KS_loss()

        return make_KS_loss(int(values[1]))
    elif loss_function == "coss":
        ngpu = len(get_available_gpus())

        def coss(yt, yp):
            lf = tf.keras.losses.CosineSimilarity(
                reduction=tf.keras.losses.Reduction.NONE
            )
            loss = lf(tf.expand_dims(yt, -1), tf.expand_dims(yp, -1))
            loss = tf.reduce_mean(loss) * (1.0 / ngpu)
            return loss

        return coss

    return loss_function


def get_metrics():
    return [
        "RootMeanSquaredError",
        "MeanAbsoluteError",
    ]  # , make_FSS_loss(20, 0), make_SSIM_loss(21), make_KS_loss(21)]


def unet(
    pretrained_weights=None,
    input_size=(256, 256, 1),
    loss_function="MeanSquaredError",
    optimizer="adam",
    n_categories=None,
):
    inputs = Input(input_size)

    def conv_block(inp, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def encoder_block(inp, num_filters):
        x = conv_block(inp, num_filters)
        p = MaxPooling2D((2, 2))(x)

        return x, p

    def decoder_block(inp, skip_connections, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inp)
        x = Concatenate()([x, skip_connections])
        x = conv_block(x, num_filters)

        return x

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Force datatype of output layer to float32; float16 is not numerically
    # stable enough in this layer

    if n_categories is None:
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid", dtype="float32")(
            d4
        )
    else:
        outputs = Conv2D(
            n_categories, 1, padding="same", activation="softmax", dtype="float32"
        )(d4)

    assert n_categories is None or loss_function == "sparse_categorical_crossentropy"

    model = Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=get_loss_function(loss_function),
        metrics=get_metrics(),
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def convlstm(
    pretrained_weights=None,
    input_size=(256, 256, 1),
    loss_function="binary_crossentropy",
):
    inp = Input(shape=(None, *input_size))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(
        x
    )

    model = Model(inp, x)

    metrics = [
        "RootMeanSquaredError",
        "MeanAbsoluteError",
        "accuracy",
        "AUC",
        make_SSIM_loss(21),
        make_SSIM_loss(11),
    ]

    model.compile(
        loss=get_loss_function(loss_function),
        optimizer=keras.optimizers.Adam(),
        metrics=metrics,
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model
