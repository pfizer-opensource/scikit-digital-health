from pytest import fixture
import numpy as np
from scipy.signal import butter, sosfiltfilt


@fixture(scope="class")
def dummy_long_data():
    rng = np.random.default_rng(1357)

    # make about 15 hours of data
    t = np.arange(0, 15 * 3600, 1 / 50)

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
    t = np.arange(0, 15 * 3600, 1 / 50)

    # simulate blocks of temperature (ie 1 temp value per block of IMU samples
    block_temp = sosfiltfilt(
        butter(1, 0.33, output="sos", btype="low"),
        (rng.random(t.size // (300 * 50)) - 0.5) * 10,
    )
    temp = 29.2 + np.repeat(block_temp, 300 * 50)

    a = (rng.random((t.size, 3)) - 0.5) * 0.07
    N3 = a.shape[0] // 3
    a[:N3, 0] = 1 + (rng.random(N3) - 0.5) * 0.15
    a[N3: 2 * N3, 1] = 0.5 + (rng.random(N3) - 0.4) * 0.15
    a[2 * N3:, 2] = 0.7 + (rng.random(t.size - 2 * N3) - 0.3) * 0.12

    # rotate 1/6 segements so there are enough points around the sphere
    N6 = a.shape[0] // 6
    a[N6: N3 + N6] *= np.array([-1, -1, 1])
    a[2 * N3 + N6:] *= np.array([-1, 1, -1])

    sos = butter(1, 2 * 1 / 50, btype="low", output="sos")
    a = sosfiltfilt(sos, a, axis=0)

    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scale = np.array([1.07, 1.05, 0.991])
    offset = np.array([5.2e-4, -3.8e-5, 1.9e-6])
    temp_scale = np.array([[-5.6e-05, -4.4e-06, 3.0e-04]])

    # correction: a' = (a + offset) * scale + (temp - mean_temp_cal) * temp_scale
    acc = (a - (temp - np.mean(temp))[:, None] @ temp_scale) / scale - offset

    return t, acc, temp, scale, offset, temp_scale