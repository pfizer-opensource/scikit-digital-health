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
