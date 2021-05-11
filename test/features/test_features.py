from numpy import zeros, allclose

from skimu.features.lib import (
    Mean,
    MeanCrossRate,
    StdDev,
    Skewness,
    Kurtosis,
)


class BaseTestFeature:
    def test_1d_ndarray(self):
        pass


class TestMean:
    def test_2d(self, get_linear_accel):
        x = get_linear_accel(0.025)

        res = Mean().compute(x)

        # pretty close, but it is noisy
        assert allclose(res, [0, 0, 1], rtol=0.005, atol=0.005)


class TestMeanCrossRate:
    def test_1d(self):
        x = zeros(10)
        x[[1, 5, 6]] = 1.0
        x[[3, 7, 8, 9]] = -1.0

        res = MeanCrossRate().compute(x)
        assert res == 0.3

    def test_2d(self):
        x = zeros((2, 10))
        x[:, [1, 5, 6]] = 1.0
        x[:, [3, 7, 8, 9]] = -1.0

        res = MeanCrossRate().compute(x)
        assert allclose(res, [0.3, 0.3])
