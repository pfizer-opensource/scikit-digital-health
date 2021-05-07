from collections.abc import Iterable

import pytest

import numpy as np
from scipy.stats import skew, kurtosis

from skimu.utility import get_windowed_view
from skimu.utility.math import *


class BaseTestRolling:
    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test(self, skip):
        """
        Test various skips since there are different optimizations for different skip values
        """
        x = np.random.random(100000)
        xw = get_windowed_view(x, 150, skip)

        if isinstance(self.truth_function, Iterable):
            truth = []
            for tf, tkw in zip(self.truth_function, self.truth_kw):
                truth.append(tf(xw, axis=-1, **tkw))

            pred = self.function(x, 150, skip)

            for p, t in zip(pred, truth):
                assert np.allclose(p, t)
        else:
            truth = self.truth_function(xw, axis=-1, **self.truth_kw)

            pred = self.function(x, 150, skip)

            assert np.allclose(pred, truth)

    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test_2d(self, skip):
        """
        Test various skips since there are different optimizations for different skip values
        """
        x = np.random.random((100000, 3))
        xw = get_windowed_view(x, 150, skip)

        if isinstance(self.truth_function, Iterable):
            truth = []
            for tf, tkw in zip(self.truth_function, self.truth_kw):
                truth.append(tf(xw, axis=1, **tkw))

            pred = self.function(x, 150, skip, axis=0)

            for p, t in zip(pred, truth):
                assert np.allclose(p, t)
        else:
            truth = self.truth_function(xw, axis=1, **self.truth_kw)

            pred = self.function(x, 150, skip, axis=0)

            assert np.allclose(pred, truth)

    @pytest.mark.parametrize(
        ("in_shape", "out_shape", "kwargs"),
        (
            ((5, 500), (5, 21), {"w_len": 100, "skip": 20, "axis": -1}),
            ((500, 5), (21, 5), {"w_len": 100, "skip": 20, "axis": 0}),
            ((500,), (21,), {"w_len": 100, "skip": 20}),
            ((3, 10, 3187), (3, 10, 3015), {"w_len": 173, "skip": 1, "axis": -1}),
        ),
    )
    def test_in_out_shapes(self, in_shape, out_shape, kwargs):
        x = np.random.random(in_shape)
        pred = self.function(x, **kwargs)

        if isinstance(pred, tuple):
            for p in pred:
                assert p.shape == out_shape
        else:
            assert pred.shape == out_shape

    def test_window_length_shape_error(self):
        x = np.random.random((5, 10))

        with pytest.raises(ValueError):
            self.function(x, 11, 1, axis=-1)

    def test_negative_error(self):
        x = np.random.random((100, 300))
        for args in (-1, 10), (10, -1), (-5, -5):
            with pytest.raises(ValueError):
                self.function(x, *args, axis=-1)

    @pytest.mark.segfault
    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test_segfault(self, skip):
        x = np.random.random(10000)

        for i in range(1000):
            pred = self.function(x, 150, skip)


class TestRollingMean(BaseTestRolling):
    # need staticmethod so it doesn't think that self is the first argument
    function = staticmethod(moving_mean)
    truth_function = staticmethod(np.mean)
    truth_kw = {}


class TestRollingSD(BaseTestRolling):
    function = staticmethod(moving_sd)
    truth_function = (np.std, np.mean)
    truth_kw = ({"ddof": 1}, {})


class TestRollingSkewness(BaseTestRolling):
    function = staticmethod(moving_skewness)
    truth_function = (skew, np.std, np.mean)
    truth_kw = ({"bias": True}, {"ddof": 1}, {})


class TestRollingKurtosis(BaseTestRolling):
    function = staticmethod(moving_kurtosis)
    truth_function = (kurtosis, skew, np.std, np.mean)
    truth_kw = (
        {"bias": True, "fisher": True, "nan_policy": "propagate"},
        {"bias": True},
        {"ddof": 1},
        {},
    )


class TestRollingMedian(BaseTestRolling):
    function = staticmethod(moving_median)
    truth_function = staticmethod(np.median)
    truth_kw = {}

    @pytest.mark.parametrize("skip", (1, 2, 7, 150, 300))
    def test_pad(self, skip):
        x = np.random.random(100000)
        xw = get_windowed_view(x, 150, skip)

        truth = self.truth_function(xw, axis=-1, **self.truth_kw)
        pred = self.function(x, 150, skip, pad=True)

        N = (x.size - 150) // skip + 1

        assert np.allclose(pred[:N], truth)
        assert np.all(np.isnan(pred[N:]))
