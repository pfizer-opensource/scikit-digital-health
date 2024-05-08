from pathlib import Path

from pytest import fixture

from numpy import arange, zeros, sin, pi, load, array


@fixture(scope="module")
def t_and_x():
    t = arange(0, 5.01, 0.02)
    x = zeros((t.size, 3))
    x[:, 0] += 1 + 0.75 * sin(2 * pi * 1.0 * t)
    x[:, 1] += 0.3 * sin(2 * pi * 0.95 * t)
    x[:, 2] += 0.1 * sin(2 * pi * 2.0 * t)

    return t, x


@fixture(scope="function")
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


@fixture(scope="function")
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


# NOTE: gait_input_50 moved to top level conftest.py


@fixture(scope="class")
def gait_res_50_apcwt(path_tests):
    return load(path_tests / "gait" / "data" / "gait_results_apcwt.npz")


@fixture(scope="class")
def gait_res_50_vcwt(path_tests):
    return load(path_tests / "gait" / "data" / "gait_results_vcwt.npz")


@fixture(scope="class")
def gait_input_gyro(path_tests):
    data = load(path_tests / "gait" / "data" / "gait_input2.npz")
    t = data["time"]
    acc = data["accel"]
    gyr = data["gyro"]

    return t, acc, gyr


@fixture(scope="class")
def gait_res_gyro_apcwt(path_tests):
    return load(path_tests / "gait" / "data" / "gait_results2_apcwt.npz")


@fixture(scope="class")
def gait_res_gyro_vcwt(path_tests):
    return load(path_tests / "gait" / "data" / "gait_results2_vcwt.npz")


@fixture(scope="module")
def get_bgait_samples_truth():  # boolean gait classification
    def get_stuff(case):
        starts = array([0, 150, 165, 200, 225, 400, 770, 990])
        stops = array([90, 160, 180, 210, 240, 760, 780, 1000])

        if case == 1:
            dt = 1 / 50
            time = arange(0, 1000 * dt, dt)
            n_max_sep = 25  # 0.5 seconds
            n_min_time = 75  # 1.5 seconds

            bouts = [slice(0, 90), slice(150, 240), slice(400, 780)]
        elif case == 2:
            dt = 1 / 100
            time = arange(0, 1000 * dt, dt)
            n_max_sep = 50  # 0.5 seconds
            n_min_time = 200  # 2 seconds

            bouts = [slice(400, 780)]

        elif case == 3:
            dt = 1 / 50
            time = arange(0, 1000 * dt, dt)
            n_max_sep = 75  # 1.5 seconds
            n_min_time = 5  # 0.1 seconds

            bouts = [slice(0, 240), slice(400, 780), slice(990, 1000)]
        else:
            dt = 1 / 50
            time = arange(0, 1000 * dt, dt)
            n_max_sep = 6  # 0.12 seconds
            n_min_time = 5  # 0.1 seconds

            bouts = [
                slice(0, 90),
                slice(150, 180),
                slice(200, 210),
                slice(225, 240),
                slice(400, 760),
                slice(770, 780),
                slice(990, 1000),
            ]
        return starts, stops, time, n_max_sep * dt, n_min_time * dt, bouts

    return get_stuff
