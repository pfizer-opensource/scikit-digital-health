from pytest import fixture
import numpy as np
from scipy.signal import butter, sosfiltfilt


@fixture(scope='module')
def sample_data_long():
    np.random.seed(1357)  # fix seed

    # make about 15 hours of data
    t = np.arange(0, 15 * 3600, 1/50)

    a = (np.random.random((t.size, 3)) - 0.5) * 0.07
    N3 = a.shape[0] // 3
    a[:N3, 0] = 1 + (np.random.random(N3) - 0.5) * 0.15
    a[N3:2*N3, 1] = 0.5 + (np.random.random(N3) - 0.4) * 0.15
    a[2*N3:, 2] = 0.7 + (np.random.random(t.size - 2*N3) - 0.3) * 0.12

    # rotate 1/6 segments so there are enough points around the sphere
    N6 = a.shape[0] // 6
    a[N6:N3 + N6] *= np.array([-1, -1, 1])
    a[2 * N3 + N6:] *= np.array([-1, 1, -1])

    sos = butter(1, 2 * 1 / 50, btype='low', output='sos')
    a = sosfiltfilt(sos, a, axis=0)

    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scale = np.array([1.07, 1.05, 0.991])
    offset = np.array([5.2e-4, -3.8e-5, 1.9e-6])

    # correction: a' = (a + offset) * scale
    ap = a / scale - offset

    return t, ap, scale, offset


@fixture(scope="module")
def sample_data_temp():
    np.random.seed(1357)  # fix seed

    # make about 73 hours of data
    t = np.arange(0, 73 * 3600, 1 / 50)
    block_temp = sosfiltfilt(
        butter(1, 0.33, output='sos', btype='low'),
        (np.random.random(t.size // (300 * 50)) - 0.5) * 10
    )
    temp = 29.2 + np.repeat(block_temp, 300 * 50)

    a = (np.random.random((t.size, 3)) - 0.5) * 0.07
    N3 = a.shape[0] // 3
    a[:N3, 0] = 1 + (np.random.random(N3) - 0.5) * 0.15
    a[N3:2 * N3, 1] = 0.5 + (np.random.random(N3) - 0.4) * 0.15
    a[2 * N3:, 2] = 0.7 + (np.random.random(t.size - 2 * N3) - 0.3) * 0.12

    # rotate 1/6 segements so there are enough points around the sphere
    N6 = a.shape[0] // 6
    a[N6:N3 + N6] *= np.array([-1, -1, 1])
    a[2 * N3 + N6:] *= np.array([-1, 1, -1])

    sos = butter(1, 2 * 1 / 50, btype='low', output='sos')
    a = sosfiltfilt(sos, a, axis=0)

    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scale = np.array([1.07, 1.05, 0.991])
    offset = np.array([5.2e-4, -3.8e-5, 1.9e-6])
    temp_scale = np.array([[-5.6e-05, -4.4e-06,  3.0e-04]])

    # correction: a' = (a + offset) * scale + (temp - mean_temp_cal) * temp_scale
    acc = (a - (temp - np.mean(temp))[:, None] @ temp_scale) / scale - offset

    return t, acc, temp, scale, offset, temp_scale


@fixture
def accel_with_nonwear():
    def get_sample(app_setup_crit, ship_crit):
        np.random.seed(1357)  # fix seed

        fs = 2

        # make 160 hours of data
        t = np.arange(0, 160 * 3600, 1 / fs)
        a = (np.random.random((t.size, 3)) - 0.5) * 0.02
        a[:, 0] += 1  # vertical axis

        wss = np.array([
            [0, 2],  # [    ][w-2][nw-1]   NF: setup after all passes
            [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
            [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
            [9, 70],  # w-61                NF
            [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
            [110, 140],  # w-30                NF
            [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
            [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
            [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
            [158, 159]  # [nw-3][w-1][nw-1]    F: 1 < 0.8(4)
        ]) * 3600 * fs  # convert to indices
        wss[1:, 0] += int(0.75 * 3600 * fs)  # because the way the windows overlap

        for se in wss:
            a[se[0]:se[1]] += (np.random.random((se[1] - se[0], 3)) - 0.5) * 0.5

        starts = np.array([0, 3, 6, 9, 110, 144]) * 3600 * fs
        stops = np.array([2, 5, 8, 70, 140, 149]) * 3600 * fs

        if app_setup_crit:
            starts = starts[1:]
            stops = stops[1:]
        starts = starts[stops > (ship_crit[0] * 3600 * fs)]
        stops = stops[stops > (ship_crit[0] * 3600 * fs)]

        wear = [[i, j] for i, j in zip(starts, stops)]

        return t, a, wear
    return get_sample


@fixture
def simple_nonwear_data():
    def sample(case, wskip, app_setup_crit, ship_crit):
        nh = int(60 / wskip)
        if case == 1:
            wss = np.array([
                [0, 2],      # [    ][w-2][nw-1]   NF: setup after all passes
                [3, 5],      # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [6, 8],      # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [9, 70],     # w-61                NF
                [90, 94],    # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                [110, 140],  # w-30                NF
                [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                [158, 159]   # [nw-3][w-1][nw-1]    F: 1 < 0.8(4)
            ]) * nh
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
            wss = np.array([
                [0, 2],  # [    ][w-2][nw-1]   NF: setup after all passes
                [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [9, 70],  # w-61                NF
                [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                [110, 140],  # w-30                NF
                [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                [158, 160]   # [nw-3][w-2][nw-0]    F: 2 < 0.8(3)
            ]) * nh
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
            wss = np.array([
                [1, 2],  # [nw-1][w-1][nw-1]    F: 1 < 0.8(2)
                [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [9, 70],  # w-61                NF
                [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                [110, 140],  # w-30                NF
                [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                [158, 160]  # [nw-3][w-2][nw-0]     F: 2 < 0.8(3)
            ]) * nh
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
            wss = np.array([
                [1, 2],  # [nw-1][w-1][nw-1]    F: 1 < 0.8(2)
                [3, 5],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [6, 8],  # [nw-1][w-2][nw-1]   NF: 2 !< 0.8(2)
                [9, 70],  # w-61                NF
                [90, 94],  # [nw-20][w-4][nw-16]  F: 4 < 0.3(36)
                [110, 140],  # w-30                NF
                [141, 142],  # [nw-1][w-1][nw-2]    F: 1 < 0.8(3)
                [144, 149],  # [nw-2][w-5][nw-4]   NF: 5 !< 0.3(6)
                [153, 155],  # [nw-4][w-2][nw-3]    F: 2 < 0.8(7)
                [158, 159]  # [nw-3][w-1][nw-1]     F: 1 < 0.8(4)
            ]) * nh
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
            nonwear[int(stst[0]):int(stst[1])] = False

        return nonwear, starts * nh, stops * nh
    return sample
