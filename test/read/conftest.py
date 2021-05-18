from pathlib import Path

from pytest import fixture
from numpy import load


@fixture
def gnactv_file():
    cwd = Path.cwd().parts

    if cwd[-1] == "read":
        return "data/gnactv_sample.bin"
    elif cwd[-1] == "test":
        return "read/data/gnactv_sample.bin"
    elif cwd[-1] == "scikit-imu":
        return "test/read/data/gnactv_sample.bin"


@fixture
def gnactv_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "read":
        path = "data/gnactv_data.npz"
    elif cwd[-1] == "test":
        path = "read/data/gnactv_data.npz"
    elif cwd[-1] == "scikit-imu":
        path = "test/read/data/gnactv_data.npz"

    dat = load(path, allow_pickle=False)

    data = {
        i: dat[i] for i in ['accel', 'time', 'temperature', 'light']
    }
    data["day_ends"] = {(8, 12): dat["day_ends_8_12"]}

    return data
