import pytest
import numpy as np
from numpy import cos, sin


@pytest.fixture(scope="module")
def get_rotation_matrix():
    def rmat(alpha, beta, gamma):
        """
        alpha : rotation around z axis - tait-bryan angle
        beta : rotation around y axis - tait-bryan
        gamma : rotation around x axis - tair-bryan
        """
        ca = cos(alpha)
        cb = cos(beta)
        cy = cos(gamma)

        sa = sin(alpha)
        sb = sin(beta)
        sy = sin(gamma)

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
