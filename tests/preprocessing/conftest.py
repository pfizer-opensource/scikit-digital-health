from pytest import fixture
import numpy as np
from scipy.signal import butter, sosfiltfilt, square


@fixture(scope="class")
def dummy_imu_data():
    rng = np.random.default_rng(24089752)
    # make 2 hours of data
    fs = 50
    t = np.arange(0, 7200, 1 / fs)
    x = np.zeros((t.size, 3))

    # create 3 regions of data
    i1 = int(t.size / 3)
    i2 = int(t.size * 2 / 3)

    # general shape
    duty = np.maximum(np.minimum(rng.normal(scale=0.1, size=i1), 0.75), 0.25)
    base = square(t[:i1] / 225, duty=duty) * 0.5
    noise1 = rng.normal(scale=0.05, size=i1)
    noise2 = rng.normal(scale=0.05, size=i1)
    noise3 = rng.normal(scale=0.05, size=i1)

    # part 1
    x[:i1, 0] = base + 0.5 + noise1
    x[:i1, 1] = base - 0.5 + noise2
    x[:i1, 2] = noise3
    # part 2
    x[i1:i2, 0] = 0.01 * noise3
    x[i1:i2, 1] = 0.005 * noise1 + 0.005 * noise3
    x[i1:i2, 2] = 0.01 * noise2
    # part 3
    x[i2:, 1] = base + 0.5 + noise2
    x[i2:, 2] = base - 0.5 + noise3
    x[i2:, 0] = noise1

    # make some temperature data
    tt = np.arange(0, 7200, 300 / fs)
    i1t = int(tt.size / 3)
    i2t = int(tt.size * 2 / 3)
    temp = np.zeros(tt.size) + 28
    temp[:i1t] += noise1[:i1t] * 20
    temp[i2t:] += noise3[:i1t] * 20

    temp[i1t:i2t] = -np.sin(tt[:i1t] / 800) * 18 + 27
    temp[i1t:i2t] = np.maximum(temp[i1t:i2t], 20) + noise2[:i1t] * 3

    # do some filtering
    sos_acc = butter(1, 2 * 5 / 50, output="sos", btype="low")
    sos_temp = butter(4, 0.15, output="sos", btype="low")

    x = sosfiltfilt(sos_acc, x, axis=0)
    temp = sosfiltfilt(sos_temp, temp)
    temp = np.repeat(temp, 300)

    return t, x, temp, fs


@fixture(scope="class")
def dummy_long_data():
    rng = np.random.default_rng(1357)

    # make about 15 hours of data
    t = np.arange(0, int(12.5 * 3600), 1 / 50)

    a = (rng.random((t.size, 3)) - 0.5) * 0.07
    N3 = a.shape[0] // 3
    a[:N3, 0] = 1 + (rng.random(N3) - 0.5) * 0.15
    a[N3 : 2 * N3, 1] = 0.5 + (rng.random(N3) - 0.4) * 0.15
    a[2 * N3 :, 2] = 0.7 + (rng.random(t.size - 2 * N3) - 0.3) * 0.12

    # rotate 1/6 segments so that there are enough points around the sphere
    N6 = a.shape[0] // 6
    a[N6 : N3 + N6] *= np.array([-1, -1, 1])
    a[2 * N3 + N6 :] *= np.array([-1, 1, -1])

    sos = butter(1, 2 * 1 / 50, btype="low", output="sos")
    a = sosfiltfilt(sos, a, axis=0)
    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scale = np.array([1.07, 1.05, 0.991])
    offset = np.array([5.2e-4, -3.8e-5, 1.9e-6])

    # correction: a' = (a + offset) * scale
    ap = a / scale - offset

    return t, ap, scale, offset


@fixture(scope="class")
def dummy_temp_data():
    rng = np.random.default_rng(1357)

    # make about 15 hours of data
    t = np.arange(0, int(12.5 * 3600), 1 / 50)

    # simulate blocks of temperature (ie 1 temp value per block of IMU samples
    block_temp = sosfiltfilt(
        butter(1, 0.33, output="sos", btype="low"),
        (rng.random(t.size // (300 * 50)) - 0.5) * 10,
    )
    temp = 29.2 + np.repeat(block_temp, 300 * 50)

    a = (rng.random((t.size, 3)) - 0.5) * 0.07
    N3 = a.shape[0] // 3
    a[:N3, 0] = 1 + (rng.random(N3) - 0.5) * 0.15
    a[N3 : 2 * N3, 1] = 0.5 + (rng.random(N3) - 0.4) * 0.15
    a[2 * N3 :, 2] = 0.7 + (rng.random(t.size - 2 * N3) - 0.3) * 0.12

    # rotate 1/6 segements so there are enough points around the sphere
    N6 = a.shape[0] // 6
    a[N6 : N3 + N6] *= np.array([-1, -1, 1])
    a[2 * N3 + N6 :] *= np.array([-1, 1, -1])

    sos = butter(1, 2 * 1 / 50, btype="low", output="sos")
    a = sosfiltfilt(sos, a, axis=0)

    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scale = np.array([1.07, 1.05, 0.991])
    offset = np.array([5.2e-4, -3.8e-5, 1.9e-6])
    temp_scale = np.array([[-5.6e-05, -4.4e-06, 3.0e-04]])

    # correction: a' = (a + offset) * scale + (temp - mean_temp_cal) * temp_scale
    acc = (a - (temp - np.mean(temp))[:, None] @ temp_scale) / scale - offset

    return t, acc, temp, scale, offset, temp_scale


@fixture(scope="module")
def accel_with_nonwear():
    def get_sample(app_setup_crit, ship_crit):
        rng = np.random.default_rng(1357)  # fix seed

        fs = 2

        # make 160 hours of data
        t = np.arange(0, 160 * 3600, 1 / fs)
        a = (rng.random((t.size, 3)) - 0.5) * 0.02
        a[:, 0] += 1  # vertical axis

        wss = (
            np.array(
                [
                    [0, 2],  # [    ][w-2][nw-1]   NF: setup after all passes
                    [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                    [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                    [9, 70],  # w-61                NF
                    [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                    [110, 140],  # w-30                NF
                    [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                    [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                    [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                    [158, 159],  # [nw-3][w-1][nw-1]    F: 1 < 0.8(4)
                ]
            )
            * 3600
            * fs
        )  # convert to indices
        wss[1:, 0] += int(0.75 * 3600 * fs)  # because the way the windows overlap

        for se in wss:
            a[se[0] : se[1]] += (rng.random((se[1] - se[0], 3)) - 0.5) * 0.5

        starts = np.array([0, 3, 6, 9, 110, 144]) * 3600 * fs
        stops = np.array([2, 5, 8, 70, 140, 149]) * 3600 * fs

        if app_setup_crit:
            starts = starts[1:]
            stops = stops[1:]
        starts = starts[stops > (ship_crit[0] * 3600 * fs)]
        stops = stops[stops > (ship_crit[0] * 3600 * fs)]

        wear = np.concatenate((starts, stops)).reshape((2, -1)).T

        return t, a, wear

    return get_sample


@fixture(scope="module")
def simple_nonwear_data():
    def get_sample(case, wskip, app_setup_crit, ship_crit):
        nh = int(60 / wskip)
        if case == 1:
            wss = (
                np.array(
                    [
                        [0, 2],  # [    ][w-2][nw-1]   NF: setup after all passes
                        [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [9, 70],  # w-61                NF
                        [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                        [110, 140],  # w-30                NF
                        [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                        [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                        [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                        [158, 159],  # [nw-3][w-1][nw-1]    F: 1 < 0.8(4)
                    ]
                )
                * nh
            )
            """
            [0, 2],      # [    ][w-2][nw-1]   NF: setup after all passes
            [3, 5],      # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
            [6, 8],      # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
            [9, 70],     # w-61                NF
            [110, 140],  # w-30                NF
            [144, 149],  # [nw-4][w-5][nw-11]   F: 5 !< 0.3(15)
            """
            starts = np.array([0, 3, 6, 9, 110, 144])
            stops = np.array([2, 5, 8, 70, 140, 149])

            if app_setup_crit:
                starts = starts[1:]
                stops = stops[1:]
            starts = starts[stops > ship_crit[0]]
            stops = stops[stops > ship_crit[0]]

        if case == 2:
            wss = (
                np.array(
                    [
                        [0, 2],  # [    ][w-2][nw-1]   NF: setup after all passes
                        [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [9, 70],  # w-61                NF
                        [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                        [110, 140],  # w-30                NF
                        [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                        [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                        [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                        [158, 160],  # [nw-3][w-2][nw-0]    F: 2 < 0.8(3)
                    ]
                )
                * nh
            )
            """
            [0, 2],  # [    ][w-2][nw-1]   NF: setup after all passes
            [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
            [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
            [9, 70],  # w-61                NF
            [110, 140],  # w-30                NF
            [144, 149],  # [nw-4][w-5][nw-9]   NF: 5 !< 0.3(13)
            """
            starts = np.array([0, 3, 6, 9, 110, 144])
            stops = np.array([2, 5, 8, 70, 140, 149])

            if app_setup_crit:
                starts = starts[1:]
                stops = stops[1:]
            starts = starts[stops > ship_crit[0]]
            stops = stops[stops > ship_crit[0]]

            starts = starts[starts < (160 - min(ship_crit[1], 12))]
            stops = stops[starts < (160 - min(ship_crit[1], 12))]

        if case == 3:
            wss = (
                np.array(
                    [
                        [1, 2],  # [nw-1][w-1][nw-1]    F: 1 < 0.8(2)
                        [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [9, 70],  # w-61                NF
                        [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                        [110, 140],  # w-30                NF
                        [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                        [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                        [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                        [158, 160],  # [nw-3][w-2][nw-0]     F: 2 < 0.8(3)
                    ]
                )
                * nh
            )
            """
            [3, 5],  # [nw-3][w-2][nw-1]    F: 2 < 0.8(4)
            [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)  filtered after 3rd pass
            [9, 70],  # w-61                NF
            [110, 140],  # w-30                NF
            [144, 149],  # [nw-4][w-5][nw-9]   NF: 5 !< 0.3(13)
            """
            starts = np.array([9, 110, 144])
            stops = np.array([70, 140, 149])

        if case == 4:
            wss = (
                np.array(
                    [
                        [1, 2],  # [nw-1][w-1][nw-1]    F: 1 < 0.8(2)
                        [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                        [9, 70],  # w-61                NF
                        [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                        [110, 140],  # w-30                NF
                        [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                        [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                        [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                        [158, 159],  # [nw-3][w-1][nw-1]     F: 1 < 0.8(4)
                    ]
                )
                * nh
            )
            """
            [3, 5],  # [nw-3][w-2][nw-1]    F: 2 < 0.8(4)
            [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2) # filtered 3rd pass
            [9, 70],  # w-61                NF
            [110, 140],  # w-30                NF
            [144, 149],  # [nw-4][w-5][nw-11]   NF: 5 !< 0.3(15)
            """
            starts = np.array([9, 110, 144])
            stops = np.array([70, 140, 149])

        nonwear = np.ones(160 * nh, dtype=np.bool_)
        for stst in wss:
            nonwear[int(stst[0]) : int(stst[1])] = False

        wear = np.concatenate((starts, stops)).reshape((2, -1)).T

        return nonwear, wear * nh

    return get_sample
