from datetime import datetime
from importlib import resources

from pytest import fixture
import numpy as np


@fixture(scope="module")
def tso_dummy_data():
    """
    Makes dummy sleep data. Default is 24hrs at 20hz. Sleep window from 10pm to 8am.
    Temperature set at 27deg C.
    """

    def get_data(freq):
        np.random.seed(0)

        # start and end dates
        dt1970 = datetime(1970, 1, 1)
        start = (datetime(2018, 1, 1, 12) - dt1970).total_seconds()
        end = (datetime(2018, 1, 2, 12) - dt1970).total_seconds()
        time = np.arange(start, end, 1 / freq)

        # sleep period
        sleep_start = (datetime(2018, 1, 1, 22) - dt1970).total_seconds()
        sleep_end = (datetime(2018, 1, 2, 8) - dt1970).total_seconds()

        sleep_starti = np.argmin(np.abs(time - sleep_start))
        sleep_endi = np.argmin(np.abs(time - sleep_end))

        # sample data
        accel = np.random.uniform(-4, 5, (time.size, 3))
        temp = np.random.uniform(-4, 5, time.size)
        lux = np.random.uniform(0, 80, time.size)

        accel[sleep_starti:sleep_endi] *= 0.01
        accel[sleep_starti:sleep_endi, 2] += 1
        temp[:] = 27.0

        return (time, accel, temp, lux), (sleep_start, sleep_end)

    return get_data


@fixture(scope="module")
def dummy_sleep_predictions():
    return np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )


@fixture(scope="module")
def activity_index_data():
    with resources.path("sleep.test_data", "test_activity_index.h5") as file_path:
        return file_path
