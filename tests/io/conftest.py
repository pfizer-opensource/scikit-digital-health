from pathlib import Path

from pytest import fixture
from numpy import load, random, arange, repeat
import pandas as pd

from skdh import BaseProcess, handle_process_returns
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

        @handle_process_returns(results_to_kwargs=True)
        @check_input_file(".abc", check_size=False)
        def predict(self, file=None, **kwargs):
            super().predict(expect_wear=False, expect_days=False, file=file, **kwargs)

            return {"in_predict": True, "test_input": 5}

    return Rdr


@fixture
def dummy_process():
    class dummyprocess(BaseProcess):
        def __init__(self):
            super().__init__()

        @handle_process_returns(results_to_kwargs=False)
        def predict(self, *, test_input=None, **kwargs):
            return {"a": 5, "test_output": test_input * 2.0}

    return dummyprocess


@fixture
def gnactv_file(path_tests):
    return str(path_tests / "io" / "data" / "gnactv_sample.bin")


@fixture
def gnactv_truth(path_tests):
    dat = load(path_tests / "io" / "data" / "gnactv_data.npz", allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time", "temperature", "light"]}
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data


@fixture
def ax3_file(path_tests):
    return str(path_tests / "io" / "data" / "ax3_sample.cwa")


@fixture
def ax3_truth(path_tests):
    dat = load(path_tests / "io" / "data" / "ax3_data.npz", allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time", "temperature", "fs"]}
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data


@fixture
def ax6_file(path_tests):
    return str(path_tests / "io" / "data" / "ax6_sample.cwa")


@fixture
def ax6_truth(path_tests):
    dat = load(path_tests / "io" / "data" / "ax6_data.npz", allow_pickle=False)

    data = {i: dat[i] for i in ["accel", "time", "gyro", "temperature", "fs"]}
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data


@fixture
def apdm_file(path_tests):
    return str(path_tests / "io" / 'data' / "apdm_sample.h5")


@fixture
def dummy_csv_contents():
    def fn(drop=True):
        df = pd.DataFrame(columns=["ts", "ax", "ay", "az"])

        # generate 72hrs of data at 32hz
        hours = 72
        fs = 32.0
        rng = random.default_rng(24089752)

        n = int(hours * 3600 * fs)  # number of samples

        df[["ax", "ay", "az"]] = rng.normal(size=(n, 3))

        start = "2020-06-06 12:00:00.000"
        tdelta = repeat(pd.to_timedelta(arange(0, hours * 3600), unit="s"), int(fs))

        df["_datetime_"] = pd.to_datetime(start) + tdelta

        # drop a chunk of data out
        if drop:
            i1 = int(13 * 3600 * fs)
            i2 = int(19 * 3600 * fs)

            df2 = df.drop(index=range(i1, i2)).reset_index(drop=True)

            return df2, fs, n
        else:
            return df, fs, n

    return fn
