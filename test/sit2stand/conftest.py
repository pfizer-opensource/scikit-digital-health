from pytest import fixture
from numpy import ones, arange


@fixture
def dummy_stillness_data(np_rng):
    x = ones(500)
    x[:300] += np_rng.standard_normal(300) * 0.5
    x[300:400] += np_rng.standard_normal(100) * 0.003
    x[400:] += np_rng.standard_normal(100) * 0.5

    return x
