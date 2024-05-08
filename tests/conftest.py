from collections.abc import Sequence
from pathlib import Path

import pytest
from numpy import zeros, arange, ndarray, sin, pi, float64, load
from numpy.random import default_rng

from skdh.utility.internal import apply_resample


def pytest_addoption(parser):
    parser.addoption(
        "--run_segfault",
        action="store_true",
        default=False,
        help="run segfault tests for extensions",
    )
    parser.addoption(
        "--run_slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "segfault: mark test as segfault testing")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run_segfault"):
        skip_segfault = pytest.mark.skip(reason="need --run_segfault option to run")
        for item in items:
            if "segfault" in item.keywords:
                item.add_marker(skip_segfault)

    if not config.getoption("--run_slow"):
        skip_slow = pytest.mark.skip(reason="need --run_slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="package")
def np_rng():
    return default_rng()


@pytest.fixture(scope="module")
def get_linear_accel(np_rng):
    def get_la(scale):
        x = zeros((3, 500), dtype=float64)
        x[2] = 1

        x += scale * np_rng.standard_normal((3, 500), dtype=float64)

        return x

    return get_la


@pytest.fixture(scope="module")
def get_cubic_signal(np_rng):
    def get_sig(a, b, c, d, scale):
        x = arange(0, 5, 0.01, dtype=float64)

        if isinstance(a, ndarray):
            a = a.reshape((-1, 1))
            b = b.reshape((-1, 1))
            c = c.reshape((-1, 1))
            d = d.reshape((-1, 1))
            scale = scale.reshape((-1, 1))

        y = (
            a * x**3
            + b * x**2
            + c * x
            + d
            + scale * np_rng.standard_normal(500, dtype=float64)
        )
        return y

    return get_sig


@pytest.fixture(scope="module")
def get_sin_signal(np_rng):
    def get_sig(a, f, scale=0.0):
        x = arange(0, 5, 0.01, dtype=float64)
        if isinstance(a, Sequence):
            y = zeros(500, dtype=float64)
            for amp, freq in zip(a, f):
                y += amp * sin(2 * pi * freq * x)
        else:
            y = a * sin(2 * pi * f * x)

        y += np_rng.standard_normal(500, dtype=float64) * scale
        return 1 / 0.01, y

    return get_sig


# ===============
# DATA
# ===============
@pytest.fixture(scope="class")
def path_tests(request):
    test_path = None
    for par in Path(request.module.__file__).parents:
        if par.stem == "tests":
            test_path = par
            break

    return test_path


@pytest.fixture(scope="class")
def gait_input_50(path_tests):
    path = path_tests / "gait" / "data" / "gait_input.npz"

    data = load(path)
    t = data["time"]
    acc = data["accel"]

    t50, (acc50,) = apply_resample(
        goal_fs=50.0, time=t, data=(acc,), indices=(), aa_filter=True
    )

    return t50, acc50
