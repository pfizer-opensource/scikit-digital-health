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
def s2s_input():
    cwd = Path.cwd().parts

    if cwd[-1] == "sit2stand":
        path = Path("data/s2s_input_data.npz")
    elif cwd[-1] == "test":
        path = Path("sit2stand/data/s2s_input_data.npz")
    elif cwd[-1] == "scikit-digital-health":
        path = Path("test/sit2stand/data/s2s_input_data.npz")

    return load(path, allow_pickle=False)


@fixture
def stillness_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "sit2stand":
        path = "data/s2s_stillness_results.npz"
    elif cwd[-1] == "test":
        path = "sit2stand/data/s2s_stillness_results.npz"
    elif cwd[-1] == "scikit-digital-health":
        path = "test/sit2stand/data/s2s_stillness_results.npz"

    data = load(path, allow_pickle=False)

    return data


@fixture
def displacement_truth():
    cwd = Path.cwd().parts

    if cwd[-1] == "sit2stand":
        path = "data/s2s_displacement_results.npz"
    elif cwd[-1] == "test":
        path = "sit2stand/data/s2s_displacement_results.npz"
    elif cwd[-1] == "scikit-digital-health":
        path = "test/sit2stand/data/s2s_displacement_results.npz"

    data = load(path, allow_pickle=False)

    return data
