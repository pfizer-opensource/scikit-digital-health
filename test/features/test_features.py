from numpy import zeros, allclose, array

from skimu.features.lib import (
    Mean,
    MeanCrossRate,
    StdDev,
    Skewness,
    Kurtosis,
)


def test_Mean(get_linear_accel):
    x = get_linear_accel(0.025)

    res = Mean().compute(x)

    # pretty close, but it is noisy
    assert allclose(res, [0, 0, 1], rtol=0.005, atol=0.005)


def test_MeanCrossRate():
    x = zeros((2, 10))
    x[:, [1, 5, 6]] = 1.0
    x[:, [3, 7, 8, 9]] = -1.0

    res = MeanCrossRate().compute(x)
    assert allclose(res, [0.3, 0.3])


def test_StdDev(get_linear_accel):
    x = get_linear_accel(0.25)

    res = StdDev().compute(x)

    # stddev should be approaching 0.25, but with only 500 samples, need to
    # allow more wiggle room
    assert allclose(res, 0.25, atol=0.02)


def test_Skewness(get_cubic_accel):
    x = get_cubic_accel(2., 1., 1., 1., 0.)

    res = Skewness().compute(x)

    assert allclose(res, 1.0, rtol=0.01)


def test_Kurtosis(get_cubic_accel):
    x = get_cubic_accel(2., 1., 1., 1., 0.)

    res = Kurtosis().compute(x)

    assert allclose(res, -0.1897956, rtol=1e-5)
