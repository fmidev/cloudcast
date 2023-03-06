import tensorflow as tf

# CIRA GUIDE TO CUSTOM LOSS FUNCTIONS FOR NEURAL
# NETWORKS IN ENVIRONMENTAL SCIENCES - VERSION 1
#
# Function to calculate "fractions skill score" (FSS).
#
# Function can be used as loss function or metric in neural networks.
#
# Implements FSS formula according to original FSS paper:
# N.M. Roberts and H.W. Lean, "Scale-Selective Verification of
# Rainfall Accumulation from High-Resolution Forecasts of Convective Events",
# Monthly Weather Review, 2008.
# This paper is referred to as [RL08] in the code below.


def make_FSS_loss(
    mask_size, bins=None, hard_discretization=False
):  # choose any mask size for calculating densities
    def norm(x):
        # normalise x to range [-1,1]
        min_ = tf.math.reduce_min(x)
        max_ = tf.math.reduce_max(x)

        if min_ == 1.0 and max_ == 1.0:
            return x

        nom = (x - min_) * 2.0
        denom = max_ - min_
        return nom / (denom + tf.keras.backend.epsilon()) - 1.0

    def my_FSS_loss(y_true, y_pred, binval, hard_discretization):
        # First: DISCRETIZE y_true and y_pred to have only binary values 0/1
        # (or close to those for soft discretization)

        # This example assumes that y_true, y_pred have the shape (None, N, N, 1).

        # Round true value and prediction into nearest single decimal

        if hard_discretization:
            y_true_binary = tf.where(
                tf.math.logical_and(y_true >= binval[0], y_true < binval[1]), 1.0, 0.0
            )
            y_pred_binary = tf.where(
                tf.math.logical_and(y_pred >= binval[0], y_pred < binval[1]), 1.0, 0.0
            )
        else:
            # Soft discretization

            y_true_mask = tf.where(
                tf.math.logical_and(y_true >= binval[0], y_true < binval[1]), 1.0, 0.0
            )
            y_true_binary = y_true * y_true_mask
            y_true_binary = tf.clip_by_value(y_true_binary, 0, 1)

            y_pred_mask = tf.where(
                tf.math.logical_and(y_pred >= binval[0], y_pred < binval[1]), 1.0, 0.0
            )
            y_pred_binary = y_pred * y_pred_mask
            y_pred_binary = tf.clip_by_value(y_pred_binary, 0, 1)

        # If neither y_true nor y_pred have values in this bin, fss is undetermined
        if tf.reduce_sum(y_true_binary) == 0 and tf.reduce_sum(y_pred_binary) == 0:
            return tf.constant(float("NaN"))

        if hard_discretization is False:
            c = 6  # make sigmoid function steep
            y_true_binary = tf.math.sigmoid(c * norm(y_true_binary))
            y_pred_binary = tf.math.sigmoid(c * norm(y_pred_binary))

        # Done with discretization.
        # To calculate densities: apply average pooling to y_true.
        # Result is O(mask_size)(i,j) in Eq. (2) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (2).
        pool1 = tf.keras.layers.AveragePooling2D(
            pool_size=(mask_size, mask_size), strides=(1, 1), padding="valid"
        )
        y_true_density = pool1(y_true_binary)

        # Need to know for normalization later how many pixels there are after pooling
        n_density_pixels = tf.cast(
            (tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32
        )

        # To calculate densities: apply average pooling to y_pred.
        # Result is M(mask_size)(i,j) in Eq. (3) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (3).

        pool2 = tf.keras.layers.AveragePooling2D(
            pool_size=(mask_size, mask_size), strides=(1, 1), padding="valid"
        )
        y_pred_density = pool2(y_pred_binary)

        # This calculates MSE(n) in Eq. (5) of [RL08].
        # Since we use MSE function, this automatically includes the factor 1/(Nx*Ny) in Eq. (5).
        MSE_n = tf.keras.losses.MeanSquaredError()(y_true_density, y_pred_density)

        # To calculate MSE_n_ref in Eq. (7) of [RL08] efficiently:
        # multiply each image with itself to get square terms, then sum up those terms.
        # Part 1 - calculate sum( O(n)i,j^2
        # Take y_true_densities as image and multiply image by itself.
        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])

        # Flatten result, to make it easier to sum over it.
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)

        # Calculate sum over all terms.
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        # Same for y_pred densitites:
        # Multiply image by itself
        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])

        # Flatten result, to make it easier to sum over it.
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)

        # Calculate sum over all terms.
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)
        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        # FSS score according to Eq. (6) of [RL08].
        # FSS = 1 - (MSE_n / MSE_n_ref)
        # FSS is a number between 0 and 1, with maximum of 1 (optimal value).
        # In loss functions: We want to MAXIMIZE FSS (best value is 1),
        # so return only the last term to minimize.
        # Avoid division by zero if MSE_n_ref == 0
        # MSE_n_ref = 0 only if both input images contain only zeros.
        # In that case both images match exactly, i.e. we should return 0.
        return MSE_n / (MSE_n_ref + tf.keras.backend.epsilon())

    @tf.function
    def run_loss(y_true, y_pred, bins=bins, hard_discretization=hard_discretization):
        if bins is None:
            bins = tf.constant(
                [[0, 0.15], [0.15, 0.85], [0.85, 1.01]], dtype=y_true.dtype
            )

        loss = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=True,
        )
        i = 0

        for i in range(bins.shape[0]):
            binval = bins[i]
            assert binval.shape[0] == 2
            lossv = my_FSS_loss(y_true, y_pred, binval, hard_discretization)
            loss = loss.write(i, lossv)

        loss = loss.stack()

        # assert loss.shape[0] == bins.shape[0]

        if hard_discretization is False:
            loss = tf.boolean_mask(loss, tf.math.is_finite(loss))
            return tf.reduce_mean(loss)

        return loss

    my_FSS_loss.__name__ = "FSS_mask_size-{}".format(mask_size)

    return run_loss
