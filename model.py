import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate, ConvLSTM2D, BatchNormalization, Conv3D, Activation
from tensorflow.keras.models import Model


def unet(pretrained_weights=None, input_size=(256,256,1), loss_function='MeanSquaredError', optimizer='adam', categories=None):

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

    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs)
    model.compile(optimizer = optimizer, loss = loss_function, metrics = ['RootMeanSquaredError','MeanAbsoluteError','accuracy'])

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

    if loss_function == "ssim":
        model.compile(loss=ssim_loss, optimizer=keras.optimizers.Adam(), metrics=[ssim_loss, 'accuracy', 'MeanAbsoluteError'])
    else:
        model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(), metrics=['accuracy', 'MeanAbsoluteError'])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model



# ssim loss function
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
