from PfyMU.features import *

from PfyMU.tests.features.conftest import TestFeature


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


class TestSignalEntropy(TestFeature):
    feature = SignalEntropy()


class TestSampleEntropy(TestFeature):
    feature = SampleEntropy(m=4, r=1.0)


class TestPermutationEntropy(TestFeature):
    feature = PermutationEntropy(order=4, delay=1, normalize=False)
