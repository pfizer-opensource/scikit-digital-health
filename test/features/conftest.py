from tempfile import NamedTemporaryFile

import pytest
from numpy import zeros, arange, ndarray


@pytest.fixture(scope="class")
def temp_bank_file():
    file = NamedTemporaryFile(mode="w")

    yield file.name

    file.close()


@pytest.fixture(scope="module")
def get_linear_accel(np_rng):
    def get_la(scale):
        x = zeros((3, 500))
        x[2] = 1

        x += scale * np_rng.standard_normal((3, 500))

        return x

    return get_la


@pytest.fixture(scope="module")
def get_cubic_accel(np_rng):
    def get_ca(a, b, c, d, scale):
        x = arange(0, 5, 0.01)

        if isinstance(a, ndarray):
            a = a.reshape((-1, 1))
            b = b.reshape((-1, 1))
            c = c.reshape((-1, 1))
            d = d.reshape((-1, 1))
            scale = scale.reshape((-1, 1))

        y = a * x**3 + b * x**2 + c * x + d + scale * np_rng.standard_normal(500)
        return y

    return get_ca
