import pytest
import numpy as np
import tensorflow as tf
from ks import make_KS_loss


def test_ks_perfect():
    arr = np.linspace(0, 1, 25).reshape(1, 5, 5, 1)
    lf = make_KS_loss(3)

    l = lf(arr, arr).numpy()
    print('perfect: {}'.format(l))
    assert l == 0


def test_ks_skilless():
    yt = np.arange(9, dtype=np.single).reshape(1, 3, 3, 1)
    yp = np.zeros(9, dtype=np.single).reshape(1, 3, 3, 1)

    lf = make_KS_loss(3)

    l = lf(yt, yp).numpy()
    print('skilless: {:.5f}'.format(l))
    assert l == pytest.approx(0.8888888)


def test_ks_random():
    # two normally distributed samples
    n = 15
    yt = np.random.normal(size=(n,n)).reshape(1,n,n,1)
    yp = np.random.normal(size=(n,n)).reshape(1,n,n,1)
    lf = make_KS_loss(5)

    l = lf(yt, yp).numpy()
    print('random: {:.5f}'.format(l))
    assert l > 0.15 and l < 0.26
 
def test_ks_mask_size():
     # two normally distributed samples
    n = 20
    yt = np.random.normal(size=(n,n)).reshape(1,n,n,1)
    yp = np.random.normal(size=(n,n)).reshape(1,n,n,1)

    for m in [3,7,11,15]:
        lf = make_KS_loss(m)
        l = lf(yt, yp).numpy()

        print('mask: {} loss: {:.5f}'.format(m, l))


if __name__ == "__main__":
    test_ks_perfect()
    test_ks_skilless()
    test_ks_random()
    test_ks_mask_size()
