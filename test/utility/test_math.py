from collections.abc import Iterable

import pytest
from numpy import allclose, mean, std, median, all, isnan
from scipy.stats import skew, kurtosis

from skdh.utility.windowing import get_windowed_view
from skdh.utility.math import (
    moving_mean,
    moving_sd,
    moving_skewness,
    moving_kurtosis,
    moving_median,
)


class BaseMovingStatsTester:
    function = staticmethod(lambda x: None)
    truth_function = staticmethod(lambda x: None)
    truth_kw = {}

    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test(self, skip, np_rng):
        x = np_rng.random(2000)
        xw = get_windowed_view(x, 150, skip)

        if isinstance(self.truth_function, Iterable):
            truth = []
            for tf, tkw in zip(self.truth_function, self.truth_kw):
                truth.append(tf(xw, axis=-1, **tkw))

            pred = self.function(x, 150, skip)

            for p, t in zip(pred, truth):
                assert allclose(p, t)
        else:
            truth = self.truth_function(xw, axis=-1, **self.truth_kw)

            pred = self.function(x, 150, skip)

            assert allclose(pred, truth)

    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test_2d(self, skip, np_rng):
        x = np_rng.random((2000, 3))
        xw = get_windowed_view(x, 150, skip)

        if isinstance(self.truth_function, Iterable):
            truth = []
            for tf, tkw in zip(self.truth_function, self.truth_kw):
                truth.append(tf(xw, axis=1, **tkw))

            pred = self.function(x, 150, skip, axis=0)
            pred1 = self.function(x, 150, skip, axis=0, return_previous=False)

            for p, t in zip(pred, truth):
                assert allclose(p, t)

            assert allclose(pred1, truth[0])
        else:
            truth = self.truth_function(xw, axis=1, **self.truth_kw)

            pred = self.function(x, 150, skip, axis=0)

            assert allclose(pred, truth)

    @pytest.mark.parametrize(
        ("in_shape", "out_shape", "kwargs"),
        (
            ((5, 500), (5, 21), {"w_len": 100, "skip": 20, "axis": -1}),
            ((500, 5), (21, 5), {"w_len": 100, "skip": 20, "axis": 0}),
            ((500,), (21,), {"w_len": 100, "skip": 20}),
            ((3, 10, 3187), (3, 10, 3015), {"w_len": 173, "skip": 1, "axis": -1}),
        ),
    )
    def test_in_out_shapes(self, in_shape, out_shape, kwargs, np_rng):
        x = np_rng.random(in_shape)
        pred = self.function(x, **kwargs)

        if isinstance(pred, tuple):
            for p in pred:
                assert p.shape == out_shape
        else:
            assert pred.shape == out_shape

    def test_window_length_shape_error(self, np_rng):
        x = np_rng.random((5, 10))

        with pytest.raises(ValueError):
            self.function(x, 11, 1, axis=-1)

    @pytest.mark.parametrize("args", ((-1, 10), (10, -1), (-5, -5)))
    def test_negative_error(self, args, np_rng):
        x = np_rng.random((100, 300))

        with pytest.raises(ValueError):
            self.function(x, *args, axis=-1)

    @pytest.mark.segfault
    def test_segfault(self, np_rng):
        x = np_rng.random(2000)

        for i in range(2000):
            self.function(x, 150, 3)
            self.function(x, 150, 151)


class TestMovingMean(BaseMovingStatsTester):
    function = staticmethod(moving_mean)
    truth_function = staticmethod(mean)
    truth_kw = {}


class TestMovingSD(BaseMovingStatsTester):
    function = staticmethod(moving_sd)
    truth_function = (std, mean)
    truth_kw = ({"ddof": 1}, {})


class TestMovingSkewness(BaseMovingStatsTester):
    function = staticmethod(moving_skewness)
    truth_function = (skew, std, mean)
    truth_kw = ({"bias": True}, {"ddof": 1}, {})


class TestMovingKurtosis(BaseMovingStatsTester):
    function = staticmethod(moving_kurtosis)
    truth_function = (kurtosis, skew, std, mean)
    truth_kw = (
        {"bias": True, "fisher": True, "nan_policy": "propagate"},
        {"bias": True},
        {"ddof": 1},
        {},
    )


class TestMovingMedian(BaseMovingStatsTester):
    function = staticmethod(moving_median)
    truth_function = staticmethod(median)
    truth_kw = {}
