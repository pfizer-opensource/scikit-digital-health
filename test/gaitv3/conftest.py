from pathlib import Path

from pytest import fixture

from numpy import arange, zeros, sin, pi, load, array


@fixture(scope='module')
def t_and_x():
    t = arange(0, 5.01, 0.02)
    x = zeros((t.size, 3))
    x[:, 0] += 1 + 0.75 * sin(2 * pi * 1.0 * t)
    x[:, 1] += 0.3 * sin(2 * pi * 0.95 * t)
    x[:, 2] += 0.1 * sin(2 * pi * 2.0 * t)

    return t, x


@fixture(scope='module')
def gait_input_gyro():
    cwd = Path.cwd().parts

    if cwd[-1] == "gaitv3":
        path = Path("data/gait_input2.npz")
    elif cwd[-1] == "test":
        path = Path("gaitv3/data/gait_input2.npz")
    elif cwd[-1] == "scikit-digital-health":
        path = Path("test/gaitv3/data/gait_input2.npz")

    data = load(path)
    t = data["time"]
    acc = data["accel"]
    gyr = data["gyro"]

    return t, acc, gyr


@fixture
def d_gait():
    gait = {
        "IC": array([50, 100, 150, 200, 250, 300, 350, 400]),
        "FC": array([110, 165, 210, 265, 310, 365, 415, 455]),
        "FC opp foot": array([65, 110, 165, 210, 260, 315, 360, 405]),
        "m1 delta h": array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]),
        "m2 delta h": array([0.08, 0.19, 0.09, 0.18, 0.18, 0.19, 0.09, 0.095]),
        "m2 delta h prime": array([0.01] * 8),
        "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2]),
        "forward cycles": array([2, 1, 0, 2, 2, 2, 1, 0]),
    }
    return gait


@fixture
def d_gait_aux():
    t = arange(0, 20, 0.02)  # 50hz

    a = zeros((t.size, 3))
    a[:, 0] = 1 + 0.75 * sin(2 * pi * 1.0 * t)  # 1Hz
    a[:, 1] = 0.35 * sin(2 * pi * 1.0 * t)  # 1 hz
    a[:, 2] = 0.25 * sin(2 * pi * 2.0 * t + pi / 2)  # 2hz, offset

    gait_aux = {
        "accel": [a[:199], a],
        "v axis": array([0] * 8),
        "inertial data i": array([0, 0, 0, 1, 1, 1, 1, 1]),
    }

    return gait_aux
