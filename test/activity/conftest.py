from pathlib import Path

from pytest import fixture
from numpy import array, load


@fixture(scope="module")
def act_acc():
    x = array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
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
            1,
            0,
            0,
            1,
        ]
    )

    return x


@fixture(scope="function")  # function so it gets reset for each test
def act_results():
    k = [
        "wake intensity gradient",
        "wake ig intercept",
        "wake ig r-squared",
        "wake max acc 2min [g]",
        "wake MVPA 5s epoch [min]",
        "wake MVPA 6min bout [min]",
        "wake MVPA avg duration",
        "wake MVPA transition probability",
        "wake MVPA gini index",
        "wake MVPA avg hazard",
        "wake MVPA power law distribution",
    ]

    return {i: [0.0] for i in k}


@fixture(scope="module")
def dummy_frag_predictions():
    return array(
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


@fixture
def activity_res():
    cwd = Path.cwd().parts

    if cwd[-1] == "activity":
        path = Path("data/activity_results.npz")
    elif cwd[-1] == "test":
        path = Path("activity/data/activity_results.npz")
    elif cwd[-1] == "scikit-digital-health":
        path = Path("test/activity/data/activity_results.npz")

    return load(path)
