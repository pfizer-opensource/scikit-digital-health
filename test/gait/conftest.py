from pytest import fixture
from numpy import array


@fixture
def dummy_gait():
    gait = {
        "IC": array([10, 20, 30, 40, 50, 60, 70, 80]),
        "FC": array([22, 33, 42, 53, 62, 73, 83, 91]),
        "FC opp foot": array([13, 22, 33, 42, 52, 63, 72, 81]),
        "delta h": array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]),
        "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2])
    }
    return gait
