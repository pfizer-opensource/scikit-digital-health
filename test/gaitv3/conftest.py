from pathlib import Path

from pytest import fixture

from numpy import arange, zeros, sin, pi, load


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
