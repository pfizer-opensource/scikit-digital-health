from pathlib import Path

from pytest import fixture
from pandas import read_csv
from numpy import arange


@fixture(scope="module")
def ambulation_positive_data():
    cwd = Path.cwd().parts

    if cwd[-1] == "context":
        path = Path("data/test1.csv")
    elif cwd[-1] == "test":
        path = Path("context/data/test1.csv")
    elif cwd[-1] == "scikit-digital-health":
        path = Path("test/context/data/test1.csv")

    accel = read_csv(path, header=None).values
    time = arange(0, len(accel) / 20, 1 / 20)

    return time, accel


@fixture(scope="module")
def ambulation_negative_data():
    cwd = Path.cwd().parts

    if cwd[-1] == "context":
        path = Path("data/test2.csv")
    elif cwd[-1] == "test":
        path = Path("context/data/test2.csv")
    elif cwd[-1] == "scikit-digital-health":
        path = Path("test/context/data/test2.csv")

    accel = read_csv(path, header=None).values
    time = arange(0, len(accel) / 20, 1 / 20)

    return time, accel


@fixture(scope="module")
def ambulation_negative_data_50hz():
    cwd = Path.cwd().parts

    if cwd[-1] == "context":
        path = Path("data/test2.csv")
    elif cwd[-1] == "test":
        path = Path("context/data/test2.csv")
    elif cwd[-1] == "scikit-digital-health":
        path = Path("test/context/data/test2.csv")

    accel = read_csv(path, header=None).values
    time = arange(0, len(accel) / 50, 1 / 50)

    return time, accel
