import glob
import numpy as np
import argparse
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from base.fileutils import *
from base.preprocess import *
from base.plotutils import *


def parse_time(str1, str2):
    try:
        return datetime.datetime.strptime(str1, "%Y-%m-%d"), datetime.datetime.strptime(
            str2, "%Y-%m-%d"
        )
    except ValueError:
        return datetime.datetime.strptime(
            str1, "%Y-%m-%d %H:%M:%S"
        ), datetime.datetime.strptime(str2, "%Y-%m-%d %H:%M:%S")


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action="store", type=str, required=True)
    parser.add_argument("--stop_date", action="store", type=str, required=True)
    parser.add_argument("--preprocess", action="store", type=str, required=True)
    parser.add_argument("--producer", action="store", type=str, default="nwcsaf")
    parser.add_argument(
        "--param", action="store", type=str, default="effective-cloudiness"
    )
    parser.add_argument("--packing_type", action="store", type=str, default="npz")
    parser.add_argument("--dtype", action="store", type=str, default="float32")
    parser.add_argument("directory", action="store")

    args = parser.parse_args()

    args.start_date, args.stop_date = parse_time(args.start_date, args.stop_date)

    if args.packing_type not in ("npz", "npy"):
        raise Exception("Packing type must be one of: npz, npy")

    dtypes = ["float32", "float16", "uint8"]
    if args.dtype not in dtypes:
        raise Exception("dtype must be one of: {}".format(dtypes))

    return args


def create_filename(args):
    return "{}/{}-{}-{}-{}-{}-{}.{}".format(
        args.directory,
        args.producer,
        args.param,
        args.start_date.strftime("%Y%m%d"),
        args.stop_date.strftime("%Y%m%d"),
        args.preprocess,
        args.dtype,
        args.packing_type,
    )


def save_to_file(datas, times, filename):
    if filename[-3:] == "npz":
        np.savez(filename, datas, times)
        print(f"Saved to file {filename}")
    elif filename[-3:] == "npy":
        timename = filename.replace(".npy", "-times.npy")
        np.save(filename, datas)
        np.save(timename, times)
        print(f"Saved to files {filename} and {timename}")


def create_timeseries(args):
    filenames = read_filenames(
        args.start_date, args.stop_date, args.producer, args.param
    )

    if len(filenames) == 0:
        sys.exit(1)

    times = np.asarray(
        list(map(lambda x: os.path.basename(x).split("_")[0], filenames))
    )

    datas = read_gribs(
        filenames, img_size=get_img_size(args.preprocess), dtype=np.dtype(args.dtype)
    )

    print("Created data shape: {}".format(datas.shape))

    save_to_file(datas, times, create_filename(args))


def create_forecast(args):
    filenames = read_filenames(
        args.start_date, args.stop_date, args.producer, args.param
    )
    filenames.sort()

    times = np.asarray(
        list(map(lambda x: os.path.basename(x).split("-")[0], filenames))
    )

    atime = None
    atimes = []

    for t in times:
        _at, _lt = t.split("+")
        _at = datetime.datetime.strptime(_at, "%Y%m%d%H%M")

        if atime is None or atime != _at:
            atime = _at
            atimes.append([])
        _lt = _lt.strip("m")
        _h, _m = _lt.split("h")
        step = datetime.timedelta(minutes=int(_m) + 60 * int(_h))
        atimes[-1].append((atime + step).strftime("%Y%m%dT%H%M%S"))

    if args.producer == "meps":
        for t in atimes:
            if len(t) != 7:
                print(
                    "Error: expecting 7 times per forecast, got {}: {}".format(
                        len(t), t
                    )
                )
                sys.exit(1)

    atimes = np.asarray(atimes)
    datas = read_gribs(filenames, img_size=get_img_size(args.preprocess))
    # reshape to match times, ie [num_forecasts, num_leadtimes, h, w, channels]
    datas = datas.reshape((atimes.shape) + datas.shape[1:])

    print("Created data shape: {}".format(datas.shape))

    save_to_file(datas, atimes, create_filename(args))


def create_sun_elevation_angle_timeseries(args):
    times = []
    datas = []

    curdate = args.start_date
    img_size = get_img_size(args.preprocess)

    start = time.time()
    while curdate != args.stop_date:
        angle = create_sun_elevation_angle(curdate, img_size).astype(args.dtype)
        datas.append(angle)
        times.append(curdate.strftime("%Y%m%dT%H%M%S"))
        curdate += datetime.timedelta(minutes=15)

    stop = time.time()

    duration = stop - start

    print(
        "Creating {} elevation angle grids took {:.2f} seconds".format(
            len(times), duration
        )
    )

    save_to_file(
        datas,
        times,
        "{}/{}_{}_{}.{}".format(
            args.directory, args.param, img_size, args.dtype, args.packing_type
        ),
    )


if __name__ == "__main__":
    args = parse_command_line()

    if args.param == "sun_elevation_angle":
        args.producer = "cloudcast"
        create_sun_elevation_angle_timeseries(args)
    elif args.producer == "nwcsaf":
        create_timeseries(args)
    else:
        create_forecast(args)
