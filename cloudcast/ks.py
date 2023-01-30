import tensorflow as tf


def make_KS_loss(mask_size=3):  # choose any mask size for calculating densities
    @tf.function
    def D(X):
        x1 = tf.sort(X[0]) # first sample = y_true
        x2 = tf.sort(X[1]) # second sample = y_pred

        # combined samples, unordered
        x = tf.concat(X, 0)

        cdf1 = tf.cast(tf.searchsorted(x1, x, side="right"), dtype=tf.float32) / float(
            x1.shape[0]
        )
        cdf2 = tf.cast(tf.searchsorted(x2, x, side="right"), dtype=tf.float32) / float(
            x2.shape[0]
        )

        cdf_diff = tf.abs(cdf1 - cdf2)
        return tf.reduce_max(cdf_diff)

    @tf.function
    def my_KS_loss(y_true, y_pred):

        mask = [1, mask_size, mask_size, 1]
        stride = [1, mask_size, mask_size, 1]

        # extract the subareas from the grid with given mask size
        # resulting tensor shape is (1, mask, mask, number_of_patches)

        true_patches = tf.image.extract_patches(
            images=y_true,
            sizes=mask,
            strides=stride,
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        pred_patches = tf.image.extract_patches(
            images=y_pred,
            sizes=mask,
            strides=stride,
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # collapse the 2d-shape of the patches, as we don't need that structure
        # when comparing histograms
        # resulting tensor shape is (1, mask^2, number_of_patches)
        true_patches = tf.reshape(true_patches, [-1, mask_size**2])
        pred_patches = tf.reshape(pred_patches, [-1, mask_size**2])

        # for each patch, execute function D
        ks = tf.map_fn(D, (true_patches, pred_patches), fn_output_signature=tf.float32)
        ks = tf.reduce_mean(ks)

        return ks

    my_KS_loss.__name__ = "KS_mask_size-{}".format(mask_size)

    return my_KS_loss
