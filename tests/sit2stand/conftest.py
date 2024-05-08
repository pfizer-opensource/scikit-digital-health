from pathlib import Path

from pytest import fixture
from numpy import ones, load


@fixture
def dummy_stillness_data(np_rng):
    x = ones(500)
    x[:300] += np_rng.standard_normal(300) * 0.5
    x[300:400] += np_rng.standard_normal(100) * 0.003
    x[400:] += np_rng.standard_normal(100) * 0.5

    return x


@fixture
def s2s_input(path_tests):
    return load(
        path_tests / "sit2stand" / "data" / "s2s_input_data.npz", allow_pickle=False
    )


@fixture
def stillness_truth(path_tests):
    data = load(
        path_tests / "sit2stand" / "data" / "s2s_stillness_results.npz",
        allow_pickle=False,
    )

    return data


@fixture
def displacement_truth(path_tests):
    data = load(
        path_tests / "sit2stand" / "data" / "s2s_displacement_results.npz",
        allow_pickle=False,
    )

    return data
