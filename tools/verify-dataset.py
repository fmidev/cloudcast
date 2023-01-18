import numpy as np
import argparse
from dateutil import parser as dateparser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
from base.plotutils import *

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", action='store')
    parser.add_argument("--show_example", action='store_true', default=False)
    parser.add_argument("--show_histogram", action='store_true', default=False)

    args = parser.parse_args()

    return args

def print_summary_statistics(data, times):
    static=0
    miss=0
    nans=0
    infs=0
    allmax=-1e38
    allmin=1e38
    
    for i,arr in enumerate(data):
        min_ = np.min(arr)
        max_ = np.max(arr)

        if np.mean(arr) == 9999.0:
            miss += 1
            print('miss:',times[i])
        if min_ == max_:
            static += 1
            print('static min=max={}, time={}'.format(max_,times[i]))
        if np.isnan(arr).sum() > 0:
            print("data contains nans: {}".format(np.isnan(arr).sum()))
            nans += 1
        if np.isinf(arr).sum() > 0:
            print("data contains inf: {}".format(np.isnan(arr).sum()))
            infs += 1
        if allmax < max_:
            allmax = max_
        if allmin > min_:
            allmin = min_
        if min_ < -0.1: 
            print('ERR: time {} min value: {}'.format(times[i], min_))
            sys.exit(1)

    print(f"out of {data.shape[0]} grids were: missing={miss}, static={static} nans={nans} infs={infs}")
    print(f"max: {allmax}, min: {allmin}, mean: {np.mean(data)}")


def show_example(data, times):
    idx = np.random.randint(data.shape[0])
    is_forecast = (len(data.shape) == 5)

    plt.figure(1)

    if is_forecast:
        idx2 = np.random.randint(data[idx].shape[0])
        print(f'showing random grid from location {idx},{idx2} (time: {times[idx][idx2]})')
        plt.imshow(np.squeeze(data[idx][idx2]))

    else:
        print(f'showing random grid from location {idx} (time: {times[idx]})')
        plt.imshow(np.squeeze(data[idx]))

    plt.show()


def show_histogram(data):

    hists = []

    hists.append(np.average(data, axis=(1,2,3)))
    #hists.append(np.quantile(data, q=0.5, axis=(1,2,3)))
    hists.append(np.var(data, axis=(1,2,3)))

    plot_histogram(hists, ['mean', 'variance'])

    plt.show()


def find_low_mean_and_variance(data, times, mean_min, mean_max, variance_min, variance_max):
    means = np.average(data, axis=(1,2,3))
    variances = np.var(data, axis=(1,2,3))
    mean_low = np.where(means < mean_min)
    mean_high = np.where(means > mean_max)
    var_low = np.where(variances < variance_min)
    var_high = np.where(variances > variance_max)

    print('mean_low',mean_low,times[mean_low])
    print('mean_high',mean_high)
    print('var_low',var_low)
    print('var_high',var_high)


def load_npz(filename):
    dataset = np.load(args.filename)

    datas = dataset['arr_0']
    times = dataset['arr_1']

    return datas,times

def load_npy(filename):
    times_filename = '{}-times.npy'.format(filename[:-4])

    datas = np.load(filename)
    times = np.load(times_filename)

    return datas,times


def verify(args):
    print(args.filename)
    if args.filename[-3:] == 'npz':
        datas, times = load_npz(args.filename)
    else:
        assert(args.filename[-3:] == 'npy')
        datas, times = load_npy(args.filename)

    is_forecast = (len(datas.shape) == 5)
    if not is_forecast:
        assert(datas.shape[0] == times.shape[0])
        print("datas and times length match: {}".format(datas.shape[0]))
    else:
        assert(datas.shape[0:2] == times.shape[0:2])
        print("datas and times length match: {}".format(datas.shape[0:2]))
       
    print("data shape: {}".format(datas.shape))

    print_summary_statistics(datas, times)
    if args.show_example:
        show_example(datas, times)
    if args.show_histogram:
        show_histogram(datas)
#    find_low_mean_and_variance(datas, times, 0.3, 0.90, 0.025, 0.28)

if __name__ == "__main__":
    args = parse_command_line()

    verify(args)
