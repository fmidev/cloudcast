import glob
import numpy as np
import argparse
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from fileutils import *
from preprocess import *
from plotutils import *

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=True)
    parser.add_argument("--stop_date", action='store', type=str, required=True)
    parser.add_argument("--preprocess", action='store', type=str, required=True)
    parser.add_argument("--producer", action='store', type=str, default='nwcsaf')
    parser.add_argument("--param", action='store', type=str, default='effective-cloudiness')

    parser.add_argument("filename", action='store')

    args = parser.parse_args()

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args


def save_to_file(datas, times, filename):
    if filename[-3:] == 'npz':
        np.savez(filename, datas, times)
        print(f'Saved to file {filename}')
    elif filename[-3:] == 'npy':
        timename = filename.replace('.npy', '-times.npy')
        np.save(filename,  datas)
        np.save(timename, times)
        print(f'Saved to files {filename} and {timename}')

def create_timeseries(args):
    filenames = read_filenames(args.start_date, args.stop_date, args.producer, args.param)
    times = np.asarray(list(map(lambda x: os.path.basename(x).split('_')[0], filenames)))

    datas = read_gribs(filenames, img_size=get_img_size(args.preprocess))
    
    print('Created data shape: {}'.format(datas.shape))

    save_to_file(datas, times, args.filename)

def create_forecast(args):
    filenames = read_filenames(args.start_date, args.stop_date, args.producer, args.param)
    filenames.sort()

    times = np.asarray(list(map(lambda x: os.path.basename(x).split('-')[0], filenames)))

    atime = None
    atimes = []

    for t in times:
        _at, _lt = t.split('+')
        _at = datetime.datetime.strptime(_at, '%Y%m%d%H%M')

        if atime is None or atime != _at:
            atime = _at
            atimes.append([])
        _lt = _lt.strip('m')
        _h, _m = _lt.split('h')
        step = datetime.timedelta(minutes = int(_m) + 60 * int(_h))
        atimes[-1].append((atime + step).strftime('%Y%m%d%H%M%S'))

    atimes = np.asarray(atimes)
    datas = read_gribs(filenames, img_size=get_img_size(args.preprocess))
    # reshape to match times, ie [num_forecasts, num_leadtimes, h, w, channels]
    datas = datas.reshape((atimes.shape) + datas.shape[1:])

    print('Created data shape: {}'.format(datas.shape))

    save_to_file(datas, atimes, args.filename)

if __name__ == "__main__":
    args = parse_command_line()

    assert(args.filename[-3:-1] == 'np')

    if args.producer == 'nwcsaf':
        create_timeseries(args)
    else:
        create_forecast(args)
