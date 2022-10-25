import pytest
import os
# import matplotlib.pyplot as plt
from dateutil import parser
from base.preprocess import *

@pytest.mark.skip(reason="plotting")
def test_create_datetime():
    da=[]
    ya=[]
    for d in np.arange(365*24):
        #dt = datetime.datetime.strptime("2022-01-01", "%Y-%m-%d") + datetime.timedelta(days=int(d))
        dt = parser.parse("2022-01-01 00:00:00 UTC") + datetime.timedelta(hours=(int(d)))
        tod, toy = create_datetime(dt, (128,128))
        print(dt, np.max(tod), np.max(toy))
        assert(np.max(tod) == np.min(tod))
        da.append(np.max(tod))
        if dt.hour == 0:
            ya.append(np.max(toy))

    plt.subplot(2, 1, 1)
    plt.plot(da[:49], label='Day')
    plt.xlabel("Hour")
 
    plt.subplot(2, 1, 2)
    plt.plot(ya, label='Year')
#    plt.legend(loc="upper left")
    plt.xlabel("Julian day")
    plt.show()

def test_create_datetime():
    dt = parser.parse("2022-05-15 12:15:00 UTC")
    tod, toy = create_datetime(dt, (3,3))

    assert(np.max(tod) == pytest.approx(-0.06540313))
    assert(np.max(tod) == np.min(tod))
    assert(np.max(toy) == pytest.approx(-0.6815403))
    assert(np.max(toy) == np.max(toy))

def test_create_leadtime():
    onehot = create_onehot_leadtime_conditioning((3,3), 3, 1)
    assert(onehot[0][0][0] == 0)
    assert(onehot[1][0][0] == 1)
    assert(onehot[2][0][0] == 0)

    squeezed = create_squeezed_leadtime_conditioning((3,3), 12, 5)
    assert(np.max(squeezed) == pytest.approx(0.41666666))

def test_create_topo():
    os.environ['S3_HOSTNAME'] = 'lake.fmi.fi'
    os.environ['CLOUDCAST_INPUT_DIR'] = 's3://cc_archive'

    topo = create_topography_data("img_size=32x32")
    assert(topo.shape == (32,32,1))
    assert(np.max(topo) <= 1.0 and np.min(topo) >= 0.0)
