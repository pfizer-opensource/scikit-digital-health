import pytest

import numpy as np
from scipy.stats import skew, kurtosis

from skimu.utility import get_windowed_view
from skimu.utility.math import *


class BaseTestRolling:
    @pytest.mark.parametrize(
        ("in_shape", "out_shape", "kwargs"),
        (
                ((5, 500), (5, 21), {"w_len": 100, "skip": 20, "axis": -1}),
                ((500, 5), (21, 5), {"w_len": 100, "skip": 20, "axis": 0}),
                ((500,), (21,), {"w_len": 100, "skip": 20}),
                ((3, 10, 3187), (3, 10, 3015), {"w_len": 173, "skip": 1, "axis": -1}),
        )
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


class TestRollingMean(BaseTestRolling):
    # need staticmethod so it doesn't think that self is the first argument
    function = staticmethod(rolling_mean)

    def test(self):
        x = np.random.random(10000)
        xw = get_windowed_view(x, 150, 150)

        truth = np.mean(xw, axis=-1)
        pred = self.function(x, 150, 150)

        assert np.allclose(pred, truth)


class TestRollingSD(BaseTestRolling):
    function = staticmethod(rolling_sd)

    def test(self):
        x = np.random.random(10000)
        xw = get_windowed_view(x, 150, 150)

        true_mean = np.mean(xw, axis=-1)
        true_sd = np.std(xw, axis=-1, ddof=1)  # sample std dev
        pred_sd, pred_mean = self.function(x, 150, 150, return_previous=True)

        assert np.allclose(pred_sd, true_sd)
        assert np.allclose(pred_mean, true_mean)


class TestRollingSkewness(BaseTestRolling):
    function = staticmethod(rolling_skewness)

    def test(self):
        x = np.random.random(10000)
        xw = get_windowed_view(x, 150, 150)

        true_mean = np.mean(xw, axis=-1)
        true_sd = np.std(xw, axis=-1, ddof=1)  # sample std dev
        true_skew = skew(xw, axis=-1, bias=True)
        pred_skew, pred_sd, pred_mean = self.function(x, 150, 150, return_previous=True)

        assert np.allclose(pred_skew, true_skew)
        assert np.allclose(pred_sd, true_sd)
        assert np.allclose(pred_mean, true_mean)


class TestRollingKurtosis(BaseTestRolling):
    function = staticmethod(rolling_kurtosis)

    def test(self):
        x = np.random.random(10000)
        xw = get_windowed_view(x, 150, 150)

        true_mean = np.mean(xw, axis=-1)
        true_sd = np.std(xw, axis=-1, ddof=1)  # sample std dev
        true_skew = skew(xw, axis=-1, bias=True)
        true_kurt = kurtosis(xw, axis=-1, fisher=True, bias=True, nan_policy='propagate')

        pred_kurt, pred_skew, pred_sd, pred_mean = self.function(x, 150, 150, return_previous=True)

        assert np.allclose(pred_kurt, true_kurt)
        assert np.allclose(pred_skew, true_skew)
        assert np.allclose(pred_sd, true_sd)
        assert np.allclose(pred_mean, true_mean)
