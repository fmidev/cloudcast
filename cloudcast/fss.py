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
    mask_size, want_discretization=0
):  # choose any mask size for calculating densities
    want_hard_discretization = want_discretization == 2
    want_soft_discretization = want_discretization == 1

    def my_FSS_loss(y_true, y_pred):

        # First: DISCRETIZE y_true and y_pred to have only binary values 0/1
        # (or close to those for soft discretization)

        # This example assumes that y_true, y_pred have the shape (None, N, N, 1).
        cutoff = 0.1  # choose the cut off value for discretization

        if want_hard_discretization:
            # Hard discretization:
            # can use that in metric, but not in loss
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)

        elif want_soft_discretization == 1:
            # Soft discretization
            c = 10  # make sigmoid function steep
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        else:
            assert want_discretization == 0
            y_true_binary = y_true
            y_pred_binary = y_pred

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
        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)
        if want_hard_discretization:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + my_epsilon)

    my_FSS_loss.__name__ = "FSS_mask_size-{}".format(mask_size)

    return my_FSS_loss
