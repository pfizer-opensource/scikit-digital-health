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
