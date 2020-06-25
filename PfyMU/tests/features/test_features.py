from PfyMU.features import *

from PfyMU.tests.features.conftest import TestFeature


# STATISTICAL MOMENT FEATURES
class TestMean(TestFeature):
    feature = Mean()


class TestMeanCrossRate(TestFeature):
    feature = MeanCrossRate()


class TestStdDev(TestFeature):
    feature = StdDev()


class TestSkewness(TestFeature):
    feature = Skewness()


class TestKurtosis(TestFeature):
    feature = Kurtosis()


# ENTROPY FEATURES
class TestSignalEntropy(TestFeature):
    feature = SignalEntropy()


class TestSampleEntropy(TestFeature):
    feature = SampleEntropy(m=4, r=1.0)


class TestPermutationEntropy(TestFeature):
    feature = PermutationEntropy(order=4, delay=1, normalize=False)


# FREQUENCY FEATURES
class TestDominantFrequency(TestFeature):
    feature = DominantFrequency(low_cutoff=0.0, high_cutoff=12.0)


class TestDominantFrequencyValue(TestFeature):
    feature = DominantFrequencyValue(low_cutoff=0.0, high_cutoff=12.0)


class TestPowerSpectralSum(TestFeature):
    feature = PowerSpectralSum(low_cutoff=0.0, high_cutoff=12.0)


class TestSpectralFlatness(TestFeature):
    feature = SpectralFlatness(low_cutoff=0.0, high_cutoff=12.0)


class TestSpectralEntropy(TestFeature):
    feature = SpectralEntropy(low_cutoff=0.0, high_cutoff=12.0)


# MISC FEATURES
class TestComplexityInvariantDistance(TestFeature):
    feature = ComplexityInvariantDistance(normalize=True)


class TestRangeCountPercentage(TestFeature):
    feature = RangeCountPercentage(range_min=-1.0, range_max=1.0)


class TestRatioBeyondRSigma(TestFeature):
    feature = RatioBeyondRSigma(r=2.0)


# SMOOTHNESS FEATURES
class TestJerkMetric(TestFeature):
    feature = JerkMetric(normalize=True)


class TestDimensionlessJerk(TestFeature):
    feature = DimensionlessJerk(log=True, signal_type='acceleration')


class TestSPARC(TestFeature):
    feature = SPARC(padlevel=4, fc=10.0, amplitude_threshold=0.05)


# STATISTICS FEATURES
class TestRange(TestFeature):
    feature = Range()


class TestIQR(TestFeature):
    feature = IQR()


class TestRMS(TestFeature):
    feature = RMS()


class TestAutocorrelation(TestFeature):
    feature = Autocorrelation(lag=1, normalize=True)


class TestLinearSlope(TestFeature):
    feature = LinearSlope()


# WAVELET FEATURES
class TestDetailPower(TestFeature):
    feature = DetailPower(wavelet='coif4', freq_band=[1.0, 3.0])


class TestDetailPowerRatio(TestFeature):
    feature = DetailPowerRatio(wavelet='coif4', freq_band=[1.0, 3.0])
