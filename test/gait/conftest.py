from pytest import fixture
from numpy import array, zeros, arange, sin, pi


@fixture
def d_gait():
    gait = {
        "IC": array([50, 100, 150, 200, 250, 300, 350, 400]),
        "FC": array([110, 165, 210, 265, 310, 365, 415, 455]),
        "FC opp foot": array([65, 110, 165, 210, 260, 315, 360, 405]),
        "delta h": array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]),
        "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2])
    }
    return gait


@fixture
def d_gait_aux():
    t = arange(0, 20, 0.02)  # 50hz

    gait_aux = {
        "accel": zeros((t.size, 3)),
        "vert axis": 0,
        "inertial data i": array([1, 1, 1, 2, 2, 2, 2, 2])
    }

    gait_aux["accel"][:, 0] = 1 + 0.75 * sin(2 * pi * 1.0 * t)  # 1Hz
    gait_aux["accel"][:, 1] = 0.35 * sin(2 * pi * 1.0 * t)  # 1 hz
    gait_aux["accel"][:, 2] = 0.25 * sin(2 * pi * 2.0 * t + pi / 2)  # 2hz, offset

    return gait_aux
