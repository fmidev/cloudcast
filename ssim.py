import tensorflow as tf

# ssim loss function

def make_SSIM_loss(mask_size = 11, mask_sigma = 1.5, k1 = 0.01, k2 = 0.03):
    def SSIM_loss(y_true, y_pred):
        tf.debugging.assert_less_equal(tf.math.reduce_max(y_true), 1.00)

#        tf.debugging.Assert(tf.math.less_equal(tf.math.reduce_max(y_true), 1.00), y_true)
#        assert((tf.math.reduce_min(y_true)) >= 0.00)

        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=mask_size, filter_sigma=mask_sigma, k1=k1, k2=k2)) # define data range as 1.0

    SSIM_loss.__name__ = 'SSIM_mask_size-{}'.format(mask_size)

    return SSIM_loss
