import tensorflow as tf


def make_bc_l1_loss(global_batch_size):
    @tf.function
    def my_bc_l1_loss(y_true, y_pred):

        bcl = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        mael = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        yt = tf.expand_dims(y_true, axis=-1)
        yp = tf.expand_dims(y_pred, axis=-1)

        bc_loss = tf.reduce_mean(bcl(yt, yp)) * (1.0 / global_batch_size)
        mae_loss = tf.reduce_mean(mael(yt, yp)) * (1.0 / global_batch_size)

        return bc_loss + mae_loss

    my_bc_l1_loss.__name__ = "binary_crossentropy-L1"
    return my_bc_l1_loss
