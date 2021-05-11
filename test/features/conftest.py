from tempfile import NamedTemporaryFile

import pytest
from numpy import zeros


@pytest.fixture(scope="class")
def temp_bank_file():
    file = NamedTemporaryFile(mode="w")

    yield file.name

    file.close()


@pytest.fixture(scope="module")
def get_linear_accel(np_rng):
    def get_la(scale):
        x = zeros((3, 100))
        x[2] = 1

        x += scale * np_rng.standard_normal((3, 100))

        return x

    return get_la
