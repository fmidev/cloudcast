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
    parser.add_argument("filename", action='store')

    args = parser.parse_args()

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args


def create(args):
    filenames = read_filenames(args.start_date, args.stop_date, 'nwcsaf')
    times = np.asarray(list(map(lambda x: os.path.basename(x).split('_')[0], filenames)))

    datas = read_gribs(filenames, img_size=get_img_size(args.preprocess))
#    datas = preprocess_many(datas, args.preprocess)
    
    print('Created data shape: {}'.format(datas.shape))
    np.savez(args.filename, datas, times)

    print(f'Saved to file {args.filename}')

if __name__ == "__main__":
    args = parse_command_line()

    assert(args.filename[-3:] == 'npz')

    create(args)
