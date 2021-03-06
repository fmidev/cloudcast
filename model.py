import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate, ConvLSTM2D, BatchNormalization, Conv3D, Activation
from tensorflow.keras.models import Model

from fss import make_FSS_loss
from ssim import make_SSIM_loss
from ks import make_KS_loss

def get_loss_function(loss_function):
    if loss_function == 'ssim':
        return make_SSIM_loss()
    elif loss_function.startswith('fss'):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_FSS_loss()

        return make_FSS_loss(int(values[1]), int(values[2]))
    elif loss_function.startswith('ks'):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_KS_loss()

        return make_KS_loss(int(values[1]))

    return loss_function


def get_metrics():
    return ['RootMeanSquaredError','MeanAbsoluteError','accuracy', "AUC", make_FSS_loss(20, 0), make_FSS_loss(10, 0), make_SSIM_loss(21), make_SSIM_loss(11)] #, make_KS_loss(3), make_KS_loss(7)]


def unet(pretrained_weights=None, input_size=(256,256,1), loss_function='MeanSquaredError', optimizer='adam', n_categories=None):

    inputs = Input(input_size)

    def conv_block(inp, num_filters):
        x = Conv2D(num_filters, 3, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def encoder_block(inp, num_filters):
        x = conv_block(inp, num_filters)
        p = MaxPooling2D((2, 2))(x)

        return x, p

    def decoder_block(inp, skip_connections, num_filters):
        x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inp)
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

    if n_categories is None:
        outputs = Conv2D(1, 1, padding='same', activation = 'sigmoid')(d4)
    else:
        outputs = Conv2D(n_categories, 1, padding = 'same', activation = 'softmax')(d4)

    assert(n_categories is None or loss_function == 'sparse_categorical_crossentropy')

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=get_loss_function(loss_function), metrics=get_metrics())

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model



def convlstm(pretrained_weights=None, input_size=(256,256,1), loss_function='binary_crossentropy'):

    inp = Input(shape=(None, * input_size))

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
    x = Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    model = Model(inp, x)

    metrics = ['RootMeanSquaredError','MeanAbsoluteError','accuracy', "AUC", make_SSIM_loss(21), make_SSIM_loss(11)]

    model.compile(loss=get_loss_function(loss_function), optimizer=keras.optimizers.Adam(), metrics=metrics)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model
