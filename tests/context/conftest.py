from pytest import fixture
from pandas import read_csv
from numpy import arange


@fixture(scope="class")
def ambulation_positive_data(path_tests):
    accel = read_csv(
        path_tests / "context" / "data" / "ambulation_test1.csv", header=None
    ).values
    time = arange(0, len(accel) / 20, 1 / 20)

    return time, accel


@fixture(scope="class")
def ambulation_negative_data(path_tests):
    accel = read_csv(
        path_tests / "context" / "data" / "ambulation_test2.csv", header=None
    ).values
    time = arange(0, len(accel) / 20, 1 / 20)

    return time, accel


@fixture(scope="class")
def ambulation_negative_data_50hz(path_tests):
    accel = read_csv(
        path_tests / "context" / "data" / "ambulation_test2.csv", header=None
    ).values
    time = arange(0, len(accel) / 50, 1 / 50)

    return time, accel
