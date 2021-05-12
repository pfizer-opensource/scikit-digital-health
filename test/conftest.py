import pytest
from numpy import zeros
from numpy.random import default_rng


def pytest_addoption(parser):
    parser.addoption(
        "--run_segfault",
        action="store_true",
        default=False,
        help="run segfault tests for extensions",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "segfault: mark test as segfault testing")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_segfault"):
        # --run_segfault given in cli: do not skip segfault tests
        return
    skip_segfault = pytest.mark.skip(reason="need --run_segfault option to run")
    for item in items:
        if "segfault" in item.keywords:
            item.add_marker(skip_segfault)


@pytest.fixture(scope="package")
def np_rng():
    return default_rng()


@pytest.fixture(scope="module")
def get_linear_accel(np_rng):
    def get_la(scale):
        x = zeros((3, 500))
        x[2] = 1

        x += scale * np_rng.standard_normal((3, 500))

        return x

    return get_la
