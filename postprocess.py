import qnorm
import numpy as np

def quantile_normalize(raw, ref):
    assert(raw.shape == ref.shape)
    shape = raw.shape

    raw = np.expand_dims(raw.flatten().transpose(), axis=0).transpose()
    ref = ref.flatten()
    q = qnorm.quantile_normalize(raw, target=ref)
    return q.reshape(shape)

