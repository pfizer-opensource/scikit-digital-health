from pytest import fixture
from numpy import array


@fixture
def dummy_gait():
    gait = {
        "IC": array([50, 100, 150, 200, 250, 300, 350, 400]),
        "FC": array([110, 165, 210, 265, 310, 365, 415, 455]),
        "FC opp foot": array([65, 110, 165, 210, 260, 315, 360, 405]),
        "delta h": array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]),
        "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2])
    }
    return gait
