import numpy as np
import argparse
from dateutil import parser as dateparser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", action='store')

    args = parser.parse_args()

#    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
#    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args


def verify(args):
    dataset = np.load(args.filename)

    print(dataset.files)
    datas = dataset['arr_0']
    times = dataset['arr_1']

    assert(datas.shape[0] == times.shape[0])
    print("datas and times length match: {}".format(datas.shape[0]))
    print("data shape: {}".format(datas.shape))
    static=0
    miss=0
    allmax=-1e38
    allmin=1e38
    
    for i,arr in enumerate(datas):
        min_ = np.min(arr)
        max_ = np.max(arr)

        if np.mean(arr) == 9999.0:
            miss += 1
            print('miss:',times[i])
        if min_ == max_:
            static += 1
            print('static:',min_,max_,times[i])
        if allmax < max_:
            allmax = max_
        if allmin > min_:
            allmin = min_

        if min_ < -0.1: 
            print('ERR: time {} min value: {}'.format(times[i], min_))
            sys.exit(1)

    print(f"found {miss} missing, {static} static grids out of {datas.shape[0]}")
    print(f"max: {allmax}, min: {allmin}, mean: {np.mean(datas)}")

    idx = np.random.randint(datas.shape[0])

    print(f'showing random grid from location {idx} (time: {times[idx]})')

    plt.imshow(datas[idx])
    plt.show()

if __name__ == "__main__":
    args = parse_command_line()

    verify(args)
