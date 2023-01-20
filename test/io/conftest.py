from pathlib import Path

from pytest import fixture
from numpy import load, random, arange, repeat
import pandas as pd

from skdh import BaseProcess
from skdh.io.base import check_input_file


@fixture
def dummy_reader_class():
    class Rdr(BaseProcess):
        _in_pipeline = False

        def __init__(self, ext_error="warn"):
            super().__init__(
                ext_error=ext_error,
            )

            if ext_error.lower() in ["warn", "raise", "skip"]:
                self.ext_error = ext_error.lower()
            else:
                raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

        @check_input_file(".abc", check_size=False)
        def predict(self, file=None, **kwargs):
            super().predict(expect_wear=False, expect_days=False, file=file, **kwargs)

            kwargs.update({"file": file, "in_predict": True})
            return (kwargs, None) if self._in_pipeline else kwargs

    return Rdr


@fixture
def gnactv_file():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        return Path("data/gnactv_sample.bin")
    elif cwd[-1] == "test":
        return Path("io/data/gnactv_sample.bin")
    elif cwd[-1] == "scikit-digital-health":
        return Path("test/io/data/gnactv_sample.bin")


@fixture
def gnactv_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        path = "data/gnactv_data.npz"
    elif cwd[-1] == "test":
        path = "io/data/gnactv_data.npz"
    elif cwd[-1] == "scikit-digital-health":
        path = "test/io/data/gnactv_data.npz"

    dat = load(path, allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time", "temperature", "light"]}
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data


@fixture
def ax3_file():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        return Path("data/ax3_sample.cwa")
    elif cwd[-1] == "test":
        return Path("io/data/ax3_sample.cwa")
    elif cwd[-1] == "scikit-digital-health":
        return Path("test/io/data/ax3_sample.cwa")


@fixture
def ax3_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        path = "data/ax3_data.npz"
    elif cwd[-1] == "test":
        path = "io/data/ax3_data.npz"
    elif cwd[-1] == "scikit-digital-health":
        path = "test/io/data/ax3_data.npz"

    dat = load(path, allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time", "temperature", "fs"]}
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data


@fixture
def ax6_file():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        return Path("data/ax6_sample.cwa")
    elif cwd[-1] == "test":
        return Path("io/data/ax6_sample.cwa")
    elif cwd[-1] == "scikit-digital-health":
        return Path("test/io/data/ax6_sample.cwa")


@fixture
def ax6_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        path = "data/ax6_data.npz"
    elif cwd[-1] == "test":
        path = "io/data/ax6_data.npz"
    elif cwd[-1] == "scikit-digital-health":
        path = "test/io/data/ax6_data.npz"

    dat = load(path, allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time", "gyro", "temperature", "fs"]}
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data


@fixture
def gt3x_file():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        return Path("data/gt3x_sample.gt3x")
    elif cwd[-1] == "test":
        return Path("io/data/gt3x_sample.gt3x")
    elif cwd[-1] == "scikit-digital-health":
        return Path("test/io/data/gt3x_sample.gt3x")


@fixture
def gt3x_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        path = "data/gt3x_data.npz"
    elif cwd[-1] == "test":
        path = "io/data/gt3x_data.npz"
    elif cwd[-1] == "scikit-digital-health":
        path = "test/io/data/gt3x_data.npz"

    dat = load(path, allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time"]}
    data["day_ends"] = {(9, 2): dat["day_ends_9_2"]}

    return data


@fixture
def apdm_file():
    cwd = Path.cwd().parts

    if cwd[-1] == "io":
        return Path("data/apdm_sample.h5")
    elif cwd[-1] == "test":
        return Path("io/data/apdm_sample.h5")
    elif cwd[-1] == "scikit-digital-health":
        return Path("test/io/data/apdm_sample.h5")


@fixture
def dummy_csv_contents():
    def fn(drop=True):
        df = pd.DataFrame(columns=["ts", "ax", "ay", "az"])

        # generate 72hrs of data at 32hz
        hours = 72
        fs = 32.0
        rng = random.default_rng(24089752)

        n = int(hours * 3600 * fs)  # number of samples

        df[['ax', 'ay', 'az']] = rng.normal(size=(n, 3))

        start = "2020-06-06 12:00:00.000"
        tdelta = repeat(pd.to_timedelta(arange(0, hours * 3600), unit='s'), int(fs))

        df['_datetime_'] = pd.to_datetime(start) + tdelta

        # drop a chunk of data out
        if drop:
            i1 = int(13 * 3600 * fs)
            i2 = int(19 * 3600 * fs)

            df2 = df.drop(index=range(i1, i2)).reset_index(drop=True)

            return df2, fs, n
        else:
            return df, fs, n
    return fn
