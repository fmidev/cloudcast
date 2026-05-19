import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config


def _s3_endpoint_and_host():
    hostname = os.environ.get("S3_HOSTNAME", "lake.fmi.fi")

    if hostname.startswith("http://") or hostname.startswith("https://"):
        return hostname, hostname.split("://", 1)[1]

    return "https://{}".format(hostname), hostname


def get_s3_hostname():
    _, hostname = _s3_endpoint_and_host()
    return hostname


def read_filenames_from_s3(
    start_time, stop_time, producer, param="effective-cloudiness"
):
    tokens = os.environ["CLOUDCAST_INPUT_DIR"].split('/')
    prefix = '/'.join(tokens[3:]) + "/" + producer + "/"
    bucket = tokens[2]
    if prefix[0] == "/":
        prefix = prefix[1:]
    print(f"Getting object listing from s3 bucket {bucket}/{prefix}")
    endpoint_url, hostname = _s3_endpoint_and_host()
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        use_ssl=True,
        config=Config(signature_version=UNSIGNED),
    )
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    start_date = int(start_time.strftime("%Y%m%d%H%M%S"))
    stop_date = int(stop_time.strftime("%Y%m%d%H%M%S"))
    filenames = []

    for page in pages:
        for item in page["Contents"]:
            f = item["Key"]
            datetime = int(f.split("/")[-1][0:15].replace("T", ""))
            if f.find(param) != -1 and datetime >= start_date and datetime < stop_date:
                filenames.append("https://{}/{}/{}".format(hostname, bucket, f))

    print("Filter matched {} files".format(len(filenames)))
    return filenames


def write_to_s3(object_name, data, **kwargs):
    endpoint_url, _ = _s3_endpoint_and_host()
    s3 = boto3.resource(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )

    bucket_name = object_name[5:].split("/")[0]
    obj_name = "/".join(object_name[5:].split("/")[1:])

    bucket = s3.Bucket(bucket_name)

    data.seek(0)

    bucket.upload_fileobj(data, obj_name)
