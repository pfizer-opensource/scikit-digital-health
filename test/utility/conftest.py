from pytest import fixture
import numpy as np


@fixture(scope="module")
def day_ends():
    return 2000, 4000


@fixture(scope="module")
def sleep_ends():
    # treat sleep as exclusionary
    sleep_starts = {
        1: np.array([200, 1200, 2200, 4200]),
        2: np.array([200, 1800, 3800]),
        3: np.array([200, 1500, 4200])
    }
    sleep_stops = {
        1: np.array([800, 1800, 2800, 5000]),
        2: np.array([800, 2500, 4400]),
        3: np.array([200, 1900, 5000])
    }
    return sleep_starts, sleep_stops


@fixture(scope="module")
def wear_ends():
    wear_starts = np.array([0, 2300, 3000])
    wear_stops = np.array([1800, 2900, 3900])
    return wear_starts, wear_stops


@fixture(scope="module")
def true_intersect_ends():
    starts = {
        1: np.array([2800, 3000]),
        2: np.array([2500, 3000]),
        3: np.array([2300, 3000])
    }
    stops = {
        1: np.array([2900, 3900]),
        2: np.array([2900, 3800]),
        3: np.array([2900, 3900])
    }
    return starts, stops


@fixture(scope="module")
def true_sleep_only_ends():
    starts = {
        1: np.array([2000, 2800]),
        2: np.array([2500]),
        3: np.array([2000])
    }
    stops = {
        1: np.array([2200, 4000]),
        2: np.array([3800]),
        3: np.array([4000])
    }
    return starts, stops
