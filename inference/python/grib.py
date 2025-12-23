import os
import eccodes as ecc
import numpy as np
from datetime import datetime, timezone
from io import BytesIO


def save_grib_message(bio, forecast, forecast_dates, grib_options):
    T, H, W = forecast.shape

    analysistime = datetime.fromtimestamp(forecast_dates[0].item(), tz=timezone.utc)

    for i in range(T):
        forecasttime = datetime.fromtimestamp(forecast_dates[i].item(), tz=timezone.utc)
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "gridType", "lambert")
        ecc.codes_set(h, "shapeOfTheEarth", 6)
        ecc.codes_set(h, "Nx", W)
        ecc.codes_set(h, "Ny", H)
        ecc.codes_set(h, "DxInMetres", 2370000 / (W - 1))
        ecc.codes_set(h, "DyInMetres", 2670000 / (H - 1))
        ecc.codes_set(h, "jScansPositively", 1)
        ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
        ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
        ecc.codes_set(h, "Latin1InDegrees", 63.3)
        ecc.codes_set(h, "Latin2InDegrees", 63.3)
        ecc.codes_set(h, "LoVInDegrees", 15)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", 251)
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", 6)
        ecc.codes_set(h, "parameterNumber", 1)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 1) # hour
        ecc.codes_set(
            h, "forecastTime", int((forecasttime - analysistime).total_seconds() / 3600)
        )
        ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
        ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products

        if grib_options is not None:
            for gopt in grib_options.split(","):
                k, v = gopt.split("=")
                typ = "d"
                elem = k.split(":")
                if len(elem) == 2:
                    typ = elem[1]
                if typ == "d":
                    v = int(v)
                elif typ == "f":
                    v = float(v)

                ecc.codes_set(h, k, v)

        ecc.codes_set_values(h, forecast[i].flatten())
        ecc.codes_write(h, bio)
        ecc.codes_release(h)

    return bio


def save_grib(datas, dates, filepath, grib_options=None):
    assert filepath[-5:] == "grib2"
    assert datas.ndim == 4
    B, T, H, W = datas.shape

    bio = BytesIO()

    for i in range(B):
        bio = save_grib_message(bio, datas[i], dates[i], grib_options)

    if filepath[0:5] == "s3://":
        # TODO
        raise NotImplementedError("S3 writing not implemented yet")
    else:
        try:
            os.makedirs(os.path.dirname(filepath))
        except FileExistsError as e:
            pass

        with open(filepath, "wb") as fp:
            fp.write(bio.getbuffer())

    print(f"Wrote file {filepath}")

