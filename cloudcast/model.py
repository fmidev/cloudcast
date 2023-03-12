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
    Add,
    Multiply,
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


def attention_unet(
    pretrained_weights=None,
    input_size=(512, 512, 4),
    loss_function="binary_crossentropy",
    optimizer="adam",
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
        # x = conv_block(x, num_filters)
        x = Conv2D(num_filters, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization()(x)

        return x

    def attention_block(inp_g, inp_x, num_filters):
        g = Conv2D(num_filters, kernel_size=1, strides=1, padding="valid")(inp_g)
        g = BatchNormalization()(g)
        x = Conv2D(num_filters, kernel_size=1, strides=1, padding="valid")(inp_x)
        x = BatchNormalization()(x)

        psi = Add()([g, x])
        psi = Activation("relu")(psi)

        psi = Conv2D(1, kernel_size=1, strides=1, padding="valid")(psi)
        psi = BatchNormalization()(psi)
        psi = Activation("sigmoid")(psi)

        return Multiply()([inp_x, psi])

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    a1 = attention_block(d1, s4, 512)
    d1 = Concatenate()([d1, a1])
    d1 = conv_block(d1, 512)

    d2 = decoder_block(d1, s3, 256)
    a2 = attention_block(d2, s3, 256)
    d2 = Concatenate()([d2, a2])
    d2 = conv_block(d2, 256)

    d3 = decoder_block(d2, s2, 128)
    a3 = attention_block(d3, s2, 128)
    d3 = Concatenate()([d3, a3])
    d3 = conv_block(d3, 128)

    d4 = decoder_block(d3, s1, 64)
    a4 = attention_block(d4, s1, 64)
    d4 = Concatenate()([d4, a4])
    d4 = conv_block(d4, 64)

    # Force datatype of output layer to float32; float16 is not numerically
    # stable enough in this layer

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid", dtype="float32")(d4)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=get_loss_function(loss_function),
        metrics=get_metrics(),
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def resnet_unet(
    pretrained_weights=None,
    input_size=(512, 512, 4),
    loss_function="binary_crossentropy",
    optimizer="adam",
):
    inputs = Input(input_size)

    def res_block_initial(inp, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, padding="same")(x)

        y = Conv2D(num_filters, 1, padding="same")(inp)
        y = BatchNormalization()(y)

        x = Add()([x, y])

        return x

    def res_block(inp, num_filters, strides):
        x = BatchNormalization()(inp)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, strides=strides[0], padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, strides=strides[1], padding="same")(x)

        y = Conv2D(num_filters, 1, strides=strides[0], padding="same")(inp)
        y = BatchNormalization()(y)

        x = Add()([x, y])

        return x

    def decoder_block(inp, skip_connection, num_filters, strides):
        x = UpSampling2D(size=(2, 2))(inp)
        x = Concatenate(axis=-1)([x, skip_connection])
        x = res_block(x, num_filters, strides)

        return x

    s1 = res_block_initial(inputs, 64)
    s2 = res_block(s1, 128, (2, 1))
    s3 = res_block(s2, 256, (2, 1))
    s4 = res_block(s3, 512, (2, 1))

    b = res_block(s4, 1024, (2, 1))

    d1 = decoder_block(b, s4, 512, (1, 1))
    d2 = decoder_block(d1, s3, 256, (1, 1))
    d3 = decoder_block(d2, s2, 128, (1, 1))
    d4 = decoder_block(d3, s1, 64, (1, 1))

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid", dtype="float32")(d4)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=get_loss_function(loss_function),
        metrics=get_metrics(),
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model
