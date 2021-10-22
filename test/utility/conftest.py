import pytest
import numpy as np


@pytest.fixture(scope="module")
def get_rotation_matrix():
    def rmat(alpha, beta, gamma):
        """
        alpha : rotation around z axis - tait-bryan angle
        beta : rotation around y axis - tait-bryan
        gamma : rotation around x axis - tair-bryan
        """
        ca = np.cos(alpha)
        cb = np.cos(beta)
        cy = np.cos(gamma)

        sa = np.sin(alpha)
        sb = np.sin(beta)
        sy = np.sin(gamma)

        r = np.zeros((3, 3))
        r[0, 0] = ca * cb
        r[0, 1] = ca * sb * sy - sa * cy
        r[0, 2] = ca * sb * cy + sa * sy
        r[1, 0] = sa * cb
        r[1, 1] = sa * sb * sy + ca * cy
        r[1, 2] = sa * sb * cy - ca * sy
        r[2, 0] = -sb
        r[2, 1] = cb * sy
        r[2, 2] = cb * cy

        return r

    return rmat


@pytest.fixture(scope="class")
def dummy_rotated_accel(np_rng, get_rotation_matrix):
    x = np_rng.standard_normal((5000, 3))
    x *= 0.1  # make the noise small
    x[:, 2] += 1.0  # make one axis gravity

    r = get_rotation_matrix(0.1, 0.13, -0.07)

    x_rot = (r @ x.T).T

    return x, x_rot


@pytest.fixture(scope="module")
def day_ends():
    return 2000, 4000


@pytest.fixture(scope="module")
def sleep_ends():
    # treat sleep as exclusionary
    sleep_starts = {
        1: np.array([200, 1200, 2200, 4200]),
        2: np.array([200, 1800, 3800]),
        3: np.array([200, 1500, 4200]),
    }
    sleep_stops = {
        1: np.array([800, 1800, 2800, 5000]),
        2: np.array([800, 2500, 4400]),
        3: np.array([200, 1900, 5000]),
    }
    return sleep_starts, sleep_stops


@pytest.fixture(scope="module")
def wear_ends():
    wear_starts = np.array([200, 2300, 3000])
    wear_stops = np.array([1800, 2900, 3900])
    return wear_starts, wear_stops


@pytest.fixture(scope="module")
def true_intersect_ends():
    starts = {
        1: np.array([2800, 3000]),
        2: np.array([2500, 3000]),
        3: np.array([2300, 3000]),
    }
    stops = {
        1: np.array([2900, 3900]),
        2: np.array([2900, 3800]),
        3: np.array([2900, 3900]),
    }
    return starts, stops


@pytest.fixture(scope="module")
def true_sleep_only_ends():
    starts = {1: np.array([2000, 2800]), 2: np.array([2500]), 3: np.array([2000])}
    stops = {1: np.array([2200, 4000]), 2: np.array([3800]), 3: np.array([4000])}
    return starts, stops


@pytest.fixture(scope="class")
def rle_arr():
    return [0] * 5 + [1] * 3 + [0] * 4 + [1] * 7 + [0] * 2 + [1] * 6 + [0] * 1


@pytest.fixture(scope="class")
def rle_truth():
    lens = np.asarray([5, 3, 4, 7, 2, 6, 1])
    indices = np.asarray([0, 5, 8, 12, 19, 21, 27])
    vals = np.asarray([0, 1, 0, 1, 0, 1, 0])
    return lens, indices, vals


@pytest.fixture(scope="class")
def dummy_time():
    return np.arange(0, 10, 0.02)  # 50 hz, 500 samples


@pytest.fixture(scope="class")
def dummy_acc(np_rng):
    return np_rng.random((500, 3))


@pytest.fixture(scope="class")
def dummy_idx_1d():
    # original and truth value
    return np.array([0, 20, 113, 265, 481]), np.array([0, 4, 23, 53, 96])


@pytest.fixture(scope="class")
def dummy_idx_2d():
    # original and truth value
    orig = np.array([[0, 20], [113, 265], [481, 499]])
    truth = np.array([[0, 4], [23, 53], [96, 99]])
    return orig, truth


@pytest.fixture(scope="module")
def dummy_frag_predictions():
    return np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
