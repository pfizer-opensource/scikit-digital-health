import pytest
import numpy as np

from skimu.features import *


class BaseTestFeature:
    def test_1d_ndarray(self, fs, x, y, z, get_1d_truth):
        xt, yt, zt = get_1d_truth(self.feature.__class__.__name__)

        xp = self.feature.compute(x, fs=fs)
        yp = self.feature.compute(y, fs=fs)
        zp = self.feature.compute(z, fs=fs)

        assert np.allclose(xp, xt)
        assert np.allclose(yp, yt)
        assert np.allclose(zp, zt)

    def test_2d_ndarray(self, fs, acc, get_2d_truth):
        truth = get_2d_truth(self.feature.__class__.__name__)
        pred = self.feature.compute(acc, fs=fs, axis=0)

        assert np.allclose(pred, truth)

    def test_3d_ndarray(self, fs, win_acc, get_3d_truth):
        truth = get_3d_truth(self.feature.__class__.__name__)
        pred = self.feature.compute(win_acc, fs=fs, axis=1)

        assert np.allclose(pred, truth)

    def test_dataframe(self, fs, df_acc, get_dataframe_truth):
        df_truth, cols = get_dataframe_truth(self.feature.__class__.__name__)
        df_pred = self.feature.compute(df_acc, fs=fs, axis=0)

        assert np.allclose(df_pred, df_truth)


# STATISTICAL MOMENT FEATURES
class TestMean(BaseTestFeature):
    feature = Mean()


class TestMeanCrossRate(BaseTestFeature):
    feature = MeanCrossRate()


class TestStdDev(BaseTestFeature):
    feature = StdDev()


class TestSkewness(BaseTestFeature):
    feature = Skewness()


class TestKurtosis(BaseTestFeature):
    feature = Kurtosis()


# ENTROPY FEATURES
class TestSignalEntropy(BaseTestFeature):
    feature = SignalEntropy()


class TestSampleEntropy(BaseTestFeature):
    feature = SampleEntropy(m=4, r=1.0)


class TestPermutationEntropy(BaseTestFeature):
    feature = PermutationEntropy(order=4, delay=1, normalize=False)


# FREQUENCY FEATURES
class TestDominantFrequency(BaseTestFeature):
    feature = DominantFrequency(padlevel=0, low_cutoff=0.0, high_cutoff=12.0)

    @pytest.mark.parametrize("fs_", ([5], "a", (10,)))
    def test_fs_error(self, fs_):
        with pytest.raises((TypeError, ValueError)):
            self.feature.compute(None, fs_)


class TestDominantFrequencyValue(BaseTestFeature):
    feature = DominantFrequencyValue(padlevel=0, low_cutoff=0.0, high_cutoff=12.0)


class TestPowerSpectralSum(BaseTestFeature):
    feature = PowerSpectralSum(padlevel=0, low_cutoff=0.0, high_cutoff=12.0)


class TestSpectralFlatness(BaseTestFeature):
    feature = SpectralFlatness(padlevel=0, low_cutoff=0.0, high_cutoff=12.0)


class TestSpectralEntropy(BaseTestFeature):
    feature = SpectralEntropy(padlevel=0, low_cutoff=0.0, high_cutoff=12.0)


# MISC FEATURES
class TestComplexityInvariantDistance(BaseTestFeature):
    feature = ComplexityInvariantDistance(normalize=True)


class TestRangeCountPercentage(BaseTestFeature):
    feature = RangeCountPercentage(range_min=-1.0, range_max=1.0)


class TestRatioBeyondRSigma(BaseTestFeature):
    feature = RatioBeyondRSigma(r=2.0)


# SMOOTHNESS FEATURES
class TestJerkMetric(BaseTestFeature):
    feature = JerkMetric()


class TestDimensionlessJerk(BaseTestFeature):
    feature = DimensionlessJerk(log=True, signal_type="acceleration")

    def test_signal_type_error(self):
        with pytest.raises(ValueError):
            dj = DimensionlessJerk(log=True, signal_type="random signal")


class TestSPARC(BaseTestFeature):
    feature = SPARC(padlevel=4, fc=10.0, amplitude_threshold=0.05)


# STATISTICS FEATURES
class TestRange(BaseTestFeature):
    feature = Range()


class TestIQR(BaseTestFeature):
    feature = IQR()


class TestRMS(BaseTestFeature):
    feature = RMS()


class TestAutocorrelation(BaseTestFeature):
    feature = Autocorrelation(lag=1, normalize=True)


class TestLinearSlope(BaseTestFeature):
    feature = LinearSlope()


# WAVELET FEATURES
class TestDetailPower(BaseTestFeature):
    feature = DetailPower(wavelet="coif4", freq_band=[1.0, 3.0])


class TestDetailPowerRatio(BaseTestFeature):
    feature = DetailPowerRatio(wavelet="coif4", freq_band=[1.0, 3.0])
