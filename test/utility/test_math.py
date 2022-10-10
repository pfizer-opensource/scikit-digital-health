from collections.abc import Iterable

import pytest
from numpy import allclose, mean, std, median, max, min, nan, full
from scipy.stats import skew, kurtosis

from skdh.utility.windowing import get_windowed_view
from skdh.utility.math import (
    moving_mean,
    moving_sd,
    moving_skewness,
    moving_kurtosis,
    moving_median,
    moving_max,
    moving_min,
)


class BaseMovingStatsTester:
    function = staticmethod(lambda x: None)
    truth_function = staticmethod(lambda x: None)
    truth_kw = {}

    @staticmethod
    def get_truth(fn, x, xw, wlen, skip, trim, tkw):
        if trim:
            return fn(xw, axis=1, **tkw)
        else:
            tshape = list(x.shape)
            tshape[0] = int((x.shape[0] - 1) // skip + 1)
            fill = int((x.shape[0] - wlen) // skip + 1)
            truth = full(tshape, nan)
            truth[:fill] = fn(xw, axis=1, **tkw)
            return truth

    @pytest.mark.parametrize("trim", (True, False))
    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test(self, skip, trim, np_rng):
        wlen = 250  # constant
        x = np_rng.random(2000)
        xw = get_windowed_view(x, wlen, skip)

        if isinstance(self.truth_function, Iterable):
            truth = []
            for tf, tkw in zip(self.truth_function, self.truth_kw):
                truth.append(self.get_truth(tf, x, xw, wlen, skip, trim, tkw))

            pred = self.function(x, wlen, skip, trim=trim)

            for p, t in zip(pred, truth):
                assert allclose(p, t, equal_nan=True)
        else:
            truth = self.get_truth(
                self.truth_function, x, xw, wlen, skip, trim, self.truth_kw
            )

            pred = self.function(x, wlen, skip, trim=trim)

            assert allclose(pred, truth, equal_nan=True)

    @pytest.mark.parametrize("trim", (True, False))
    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test_2d(self, skip, trim, np_rng):
        wlen = 150  # constant
        x = np_rng.random((2000, 3))
        xw = get_windowed_view(x, wlen, skip)

        if isinstance(self.truth_function, Iterable):
            truth = []
            for tf, tkw in zip(self.truth_function, self.truth_kw):
                truth.append(self.get_truth(tf, x, xw, wlen, skip, trim, tkw))

            pred = self.function(x, wlen, skip, trim=trim, axis=0)
            pred1 = self.function(
                x, wlen, skip, trim=trim, axis=0, return_previous=False
            )

            for p, t in zip(pred, truth):
                assert allclose(p, t, equal_nan=True)

            assert allclose(pred1, truth[0], equal_nan=True)
        else:
            truth = self.get_truth(
                self.truth_function, x, xw, wlen, skip, trim, self.truth_kw
            )

            pred = self.function(x, wlen, skip, trim=trim, axis=0)

            assert allclose(pred, truth, equal_nan=True)

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


class TestMovingMax(BaseMovingStatsTester):
    function = staticmethod(moving_max)
    truth_function = staticmethod(max)
    truth_kw = {}


class TestMovingMin(BaseMovingStatsTester):
    function = staticmethod(moving_min)
    truth_function = staticmethod(min)
    truth_kw = {}
