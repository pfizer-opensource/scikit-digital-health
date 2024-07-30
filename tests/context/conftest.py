from pytest import fixture
from pandas import read_csv
from numpy import arange, vstack


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


@fixture(scope="class")
def motion_positive_data_100hz(path_tests):
    fs = 100
    accel = read_csv(
        path_tests / "context" / "data" / "motion_test_positive.csv"
    ).values
    accel = vstack([accel, accel, accel])
    time = arange(0, len(accel) / fs, 1 / fs)

    return time, accel, fs


@fixture(scope="class")
def motion_negative_data_100hz(path_tests):
    fs = 100
    accel = read_csv(
        path_tests / "context" / "data" / "motion_test_negative.csv"
    ).values
    time = arange(0, len(accel) / fs, 1 / fs)

    return time, accel, fs


@fixture(scope="class")
def motion_positive_data_20hz(path_tests):
    fs = 20
    accel = read_csv(
        path_tests / "context" / "data" / "motion_test_positive.csv"
    ).values
    accel = vstack([accel, accel, accel])
    time = arange(0, len(accel) / fs, 1 / fs)

    return time, accel, fs


@fixture(scope="class")
def motion_negative_data_20hz(path_tests):
    fs = 20
    accel = read_csv(
        path_tests / "context" / "data" / "motion_test_negative.csv"
    ).values
    time = arange(0, len(accel) / fs, 1 / fs)

    return time, accel, fs
