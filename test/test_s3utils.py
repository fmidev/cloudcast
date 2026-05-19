import datetime

from base import s3utils


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return self._pages


class _FakeS3Client:
    def __init__(self, pages):
        self._pages = pages

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self._pages)


def test_read_filenames_from_s3_uses_s3_hostname(monkeypatch):
    monkeypatch.setenv("CLOUDCAST_INPUT_DIR", "s3://cc_archive/source")
    monkeypatch.setenv("S3_HOSTNAME", "s3.example.test")

    pages = [
        {
            "Contents": [
                {"Key": "source/nwcsaf/20240115T114500_nwcsaf_effective-cloudiness.grib2"},
                {"Key": "source/nwcsaf/20240115T120000_nwcsaf_effective-cloudiness.grib2"},
            ]
        }
    ]

    captured = {}

    def fake_client(service, **kwargs):
        captured["service"] = service
        captured["endpoint_url"] = kwargs["endpoint_url"]
        return _FakeS3Client(pages)

    monkeypatch.setattr(s3utils.boto3, "client", fake_client)

    start = datetime.datetime.strptime("2024-01-15 11:45:00", "%Y-%m-%d %H:%M:%S")
    stop = datetime.datetime.strptime("2024-01-15 12:00:00", "%Y-%m-%d %H:%M:%S")
    filenames = s3utils.read_filenames_from_s3(start, stop, "nwcsaf")

    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "https://s3.example.test"
    assert filenames == [
        "https://s3.example.test/cc_archive/source/nwcsaf/20240115T114500_nwcsaf_effective-cloudiness.grib2"
    ]


def test_read_filenames_from_s3_accepts_s3_hostname_with_scheme(monkeypatch):
    monkeypatch.setenv("CLOUDCAST_INPUT_DIR", "s3://cc_archive/source")
    monkeypatch.setenv("S3_HOSTNAME", "https://s3.example.test")

    pages = [
        {"Contents": [{"Key": "source/nwcsaf/20240115T114500_nwcsaf_effective-cloudiness.grib2"}]}
    ]

    captured = {}

    def fake_client(service, **kwargs):
        captured["endpoint_url"] = kwargs["endpoint_url"]
        return _FakeS3Client(pages)

    monkeypatch.setattr(s3utils.boto3, "client", fake_client)

    start = datetime.datetime.strptime("2024-01-15 11:45:00", "%Y-%m-%d %H:%M:%S")
    stop = datetime.datetime.strptime("2024-01-15 12:00:00", "%Y-%m-%d %H:%M:%S")
    filenames = s3utils.read_filenames_from_s3(start, stop, "nwcsaf")

    assert captured["endpoint_url"] == "https://s3.example.test"
    assert filenames == [
        "https://s3.example.test/cc_archive/source/nwcsaf/20240115T114500_nwcsaf_effective-cloudiness.grib2"
    ]
