import pytest


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
