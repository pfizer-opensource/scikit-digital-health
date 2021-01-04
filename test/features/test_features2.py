from abc import ABC, abstractmethod

import pytest
import numpy as np
from scipy.stats import skew, kurtosis

from skimu.features import *


class _BaseTestFeature(ABC):
    atol = 1e-8  # default numpy value
    feature = None

    @abstractmethod
    def get_truth(self, sig):
        pass

    @pytest.mark.parametrize("ndim", (1, 2, 3, 4, 5))
    def test_nd(self, random_wave, ndim):
        fs = 100.

        signal = random_wave(fs, ndim)

        truth = self.get_truth(signal)

        pred = self.feature.compute(signal.y)

        assert np.allclose(pred, truth, atol=self.atol)


class TestMean:
    feature = Mean()

    @pytest.mark.parametrize("ndim", (1, 2, 3, 4, 5))
    def test_nd(self, random_linear, ndim):
        fs = 50.
        lin = random_linear(fs, ndim)

        pred = self.feature.compute(lin.y, axis=lin.axis)

        # truth
        N2 = lin.x.size // 2
        N1 = N2 - 1

        truth = (lin.slope * lin.x[..., N1] + lin.slope * lin.x[..., N2]) / 2 + lin.itcpt

        assert np.allclose(pred, truth)

    def get_random_wave_truth(self, sig):
        def integrate(y):
            p2 = 2 * np.pi
            return (-sig.amp1 / (p2 * sig.freq1) * np.cos(p2 * sig.freq1 * y)
                    - sig.amp2 / (p2 * sig.freq2) * np.cos(p2 * sig.freq1 * y)
                    + sig.slope * y**2 / 2 + sig.itcpt * y)

        return ((integrate(sig.x[..., -1]) - integrate(sig.x[..., 0]))
                / (sig.x[..., -1] - sig.x[..., 0]))


class TestMeanCrossRate:
    feature = MeanCrossRate()

    @pytest.mark.parametrize("ndim", (1, 2, 3, 4, 5))
    def test_nd(self, random_wave, random_linear, ndim):
        fs = 50.
        lin = random_linear(fs, ndim)

        pred = self.feature.compute(lin.y, axis=lin.axis)

        # truth
        truth = np.ones(pred.shape) / lin.x.size
        if np.any(lin.slope == 0):
            truth[np.nonzero(lin.slope == 0)] = 0.

        assert np.allclose(pred, truth)

        # make a custom wave with a known number of crossings
        wave = random_wave(fs, ndim)
        wave.freq1 = np.ones(wave.freq1.shape) * 0.5
        wave.amp1 = np.ones(wave.amp1.shape)
        wave.amp2 = wave.freq2 = wave.slope = np.zeros(wave.freq2.shape)
        wave.itcpt = np.zeros(wave.itcpt.shape) + 1e-5  # make sure the first value is not 0

        wave.get_y()

        pred = self.feature.compute(wave.y, axis=wave.axis)

        truth = np.ones_like(pred) * 9 / wave.x.size
        assert np.allclose(pred, truth)


class TestStdDev(_BaseTestFeature):
    feature = StdDev()

    def get_truth(self, sig):
        return np.std(sig.y, axis=sig.axis, ddof=1)


class TestSkewness(_BaseTestFeature):
    feature = Skewness()

    def get_truth(self, sig):
        return skew(sig.y, axis=sig.axis, bias=False)


class TestKurtosis(_BaseTestFeature):
    feature = Kurtosis()

    def get_truth(self, sig):
        return kurtosis(sig.y, axis=sig.axis, bias=False)


# =========================
#   Entropy Features
# =========================
class TestSignalEntropy:
    feature = SignalEntropy()

    @pytest.mark.parametrize("ndim", (1, 2, 3, 4, 5))
    def test_nd(self, ndim):
        shape = (ndim,) * (ndim - 1) + (500,)

        x = (np.random.rand(*shape) < 0.5).astype(np.float_)

        res = self.feature.compute(x)

        assert True

"""
import pytest

from skimu.features.core import Feature

from .conftest import F1, F2, F3


class TestBaseFeature:
    def test_abc(self):
        with pytest.raises(TypeError):
            Feature()

    def test_equivalence(self):
        f1_a = F1(p2=10, p1=5)
        f1_b = F1(p1=5, p2=10)[0]
        f2 = F2(p1=5, p2=10)
        f3 = F3()

        assert f1_a == f1_a
        assert f1_b == f1_b
        assert f1_a != f1_b
        assert f1_a != f2
        assert f1_b != f2
        assert f2 != f3

    def test_indexing(self):
        f = F1()

        f[0]
        assert f.n == 1
        assert f.index == 0

        f[:]
        assert f.n == -1
        assert f.index == slice(None)

        f[[0, 2]]
        assert f.n == 2
        assert f.index == [0, 2]

        f[(0, 1)]
        assert f.n == 2
        assert f.index == (0, 1)

        f[slice(2)]
        assert f.n == -1  # should be -1 since possible to not use the whole slice
        assert f.index == slice(2)

    def test_indexing_errors(self):
        f = F1()
        error = ValueError
        with pytest.raises(error):
            f[..., 2]

        with pytest.raises(error):
            f[..., [0, 2]]

        with pytest.raises(error):
            f[:, 0]

        with pytest.raises(error):
            f[:, :, 0]

        with pytest.raises(error):
            f[:, :, [0, 1]]

        with pytest.raises(error):
            f[:, :]
"""
