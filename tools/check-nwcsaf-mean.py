import numpy as np
import eccodes as ecc
import requests
import os
import boto3
import sys

from datetime import datetime, timedelta

s3client = None


def read_grib_contents(gh):
    ni = ecc.codes_get_long(gh, "Ni")
    nj = ecc.codes_get_long(gh, "Nj")

    dtype = "float32"
    data = ecc.codes_get_double_array(gh, "values").astype(dtype).reshape(nj, ni)

    if ecc.codes_get(gh, "jScansPositively"):
        data = np.flipud(data)  # image data is +x-y

    ecc.codes_release(gh)

    if np.max(data) == 9999.0 and np.min(data) == 9999.0:
        data[data == 9999.0] = np.NAN
        return np.expand_dims(data, axis=-1)

    if np.min(data) < -0.01 or np.max(data) > 1.01:
        print(
            "Invalid data found from '{}': min={} max={}".format(
                kwargs["fileuri"], np.min(data), np.max(data)
            )
        )
        sys.exit(1)

    return data


def read_from_http(url):
    if url[0:5] == "s3://":
        tokens = url[5:].split("/")
        tokens[0] = "{}/{}".format(os.environ["S3_HOSTNAME"], tokens[0])
        url = "https://" + "/".join(tokens)

    r = requests.get(url, stream=True)
    if r.status_code == 404:
        print(f"Not found: {url}")
        return None
    if r.status_code != 200:
        print(f"HTTP error: {r.status_code}")
        sys.exit(1)

    gh = ecc.codes_new_from_message(r.content)
    return read_grib_contents(gh)


def read_from_file(file_path):
    try:
        with open(file_path) as fp:
            gh = ecc.codes_new_from_file(fp, ecc.CODES_PRODUCT_GRIB)
            return read_grib_contents(gh, fileuri=file_path)
    except FileNotFoundError as e:
        print(e)


def object_exists(url):
    print("Checking for {}".format(url))
    response = requests.get(url)
    if response.status_code == 200:
        return True
    return False


def write_object(bucket, key, content):
    global s3client

    if s3client is None:
        hostname = os.environ["S3_HOSTNAME"]

        if not hostname.startswith("http"):
            hostname = "https://{}".format(hostname)

        s3client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            endpoint_url=hostname,
        )

    print("Writing {}/{}".format(bucket, key))
    s3client.put_object(Body=content, Bucket=bucket, Key=key)


def format_key(date):
    return "{}/{}/{}/{}_nwcsaf_effective-cloudiness".format(
        date.strftime("%Y"),
        date.strftime("%m"),
        date.strftime("%d"),
        date.strftime("%Y%m%dT%H%M%S"),
    )


def check(nowtime, plot=False):
    print("Now is time {}".format(nowtime.strftime("%Y%m%dT%H%M%S")))
    start = nowtime - timedelta(minutes=75)
    stop = nowtime  # datetime.strptime("20220829T010000", "%Y%m%dT%H%M%S")

    dates = []

    replace_dates = []
    while start <= stop:
        # if os.path.exists("{}.flag".format(start.strftime("%Y%m%dT%H%M%S"))):
        if object_exists("{}/{}.flag".format(urlbase, format_key(start))):
            print("Date {} is blacklisted".format(start))
            replace_dates.append(start)
            start = start + timedelta(minutes=15)
            continue
        dates.append(start)
        start = start + timedelta(minutes=15)

    datas = []

    for d in dates:
        data = read_from_http("{}/{}.grib2".format(urlbase, format_key(d)))
        if data is not None:
            datas.append(data)

    means = []

    for i in range(len(datas)):
        means.append(np.mean(datas[i]))

    if len(means) == 1:
        print("No history, unable to do check")
        return
    else:
        window_mean = np.mean(means[:-1])

    diff = np.abs(window_mean - means[-1])

    valid_dates = dates
    print(
        f"Past {len(means)-1} frame mean of means is {window_mean:.3f} current mean is {means[-1]:.3f} diff {diff:.4f}"
    )

    if diff > 0.06:
        print("{} is suspicious".format(nowtime))
        #        with open("{}.flag".format(nowtime.strftime("%Y%m%dT%H%M%S")), 'w') as fp:
        #            fp.write("{} with mean {:.3f} previous {} mean of means {:.3f} diff {:.3f}\n".format(nowtime, means[-1], len(means), window_mean, diff))
        msg = (
            "{} with mean {:.3f} previous {} mean of means {:.3f} diff {:.3f}\n".format(
                nowtime, means[-1], len(means), window_mean, diff
            )
        )
        write_object(
            "routines-data",
            "cloudcast-source/nwcsaf/{}.flag".format(format_key(nowtime)),
            msg,
        )
        valid_dates = dates[0:-1]
        replace_dates.append(dates[-1])

    print(
        "Valid dates for input are: {}".format(
            " ".join("{}".format(k.strftime("%Y%m%dT%H%M%S")) for k in valid_dates)
        )
    )

    if len(replace_dates) > 0:
        print(
            "Invalid dates for input are: {}".format(
                " ".join(
                    "{}".format(k.strftime("%Y%m%dT%H%M%S")) for k in replace_dates
                )
            )
        )

    if plot:
        import matplotlib.pyplot as plt

        img = datas[0]
        fig, axes = plt.subplots(
            nrows=1, ncols=len(means), figsize=(10, 6), sharex=True, sharey=True
        )
        ax = axes.ravel()

        ax[0].imshow(datas[0], cmap=plt.cm.gray_r, vmin=0, vmax=1)
        ax[0].set_xlabel(f"MEAN: {means[0]:.3f}")  # (f'MSE: 0, SSIM: 1')
        ax[0].set_title(dates[0].strftime("%H:%M"))

        for i in range(len(means)):
            ax[i].imshow(datas[i], cmap=plt.cm.gray_r, vmin=0, vmax=1)
            ax[i].set_xlabel(f"MEAN: {means[i]:.3f}")
            ax[i].set_title(dates[i].strftime("%H:%M"))

        plt.tight_layout()
        plt.show()


urlbase = "https://lake.fmi.fi/routines-data/cloudcast-source/nwcsaf"

if len(sys.argv) == 2:
    check(datetime.strptime(sys.argv[1], "%Y%m%dT%H%M%S"))
else:
    nowtime = datetime.strptime("20220829T224500", "%Y%m%dT%H%M%S")

    times = []
    for i in range(10):
        times.append(nowtime + timedelta(minutes=i * 15))

    for t in times:
        check(t)
