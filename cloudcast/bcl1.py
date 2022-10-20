import tensorflow as tf

def make_bc_l1_loss():

    @tf.function
    def my_bc_l1_loss(y_true, y_pred):

        bc_lossfunction = tf.keras.losses.BinaryCrossentropy()
        mae_lossfunction = tf.keras.losses.MeanAbsoluteError()


        bc_loss = bc_lossfunction(y_true, y_pred)
        mae_loss = mae_lossfunction(y_true, y_pred)

        return bc_loss + mae_loss

    my_bc_l1_loss.__name__ = 'binary_crossentropy-L1'
    return my_bc_l1_loss

