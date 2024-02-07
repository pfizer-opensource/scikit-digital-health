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
        "wake dfa alpha",
        "wake dfa activity balance index",
        "wake threshold equal avg duration [g]",
        "wake duration equal avg duration [min]",
        "wake 15min signal entropy",
        "wake 15min sample entropy",
        "wake 15min permutation entropy",
        "wake 15min power spectral sum",
        "wake 15min spectral flatness",
        "wake 15min spectral entropy",
        "wake 15min sparc",
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
def activity_res(path_tests):
    return load(path_tests / "activity" / "data" / "activity_results.npz")
