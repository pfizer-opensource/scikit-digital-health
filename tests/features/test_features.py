import pytest
from numpy import zeros, allclose, isclose, sqrt, diff, sum, std, abs, array

from skdh.features.lib import (
    Mean,
    MeanCrossRate,
    StdDev,
    Skewness,
    Kurtosis,
    DominantFrequency,
    DominantFrequencyValue,
    PowerSpectralSum,
    RangePowerSum,
    SpectralFlatness,
    SpectralEntropy,
    Range,
    IQR,
    RMS,
    Autocorrelation,
    LinearSlope,
    ComplexityInvariantDistance,
    RangeCountPercentage,
    RatioBeyondRSigma,
    SignalEntropy,
    SampleEntropy,
    PermutationEntropy,
    JerkMetric,
    DimensionlessJerk,
    SPARC,
    DetailPower,
    DetailPowerRatio,
)


def test_size_0_input():
    ext_features = [
        DominantFrequency,
        DominantFrequencyValue,
        PowerSpectralSum,
        RangePowerSum,
        SpectralFlatness,
        SpectralEntropy,
        Autocorrelation,
        LinearSlope,
        ComplexityInvariantDistance,
        RangeCountPercentage,
        RatioBeyondRSigma,
        SignalEntropy,
        SampleEntropy,
        PermutationEntropy,
        JerkMetric,
        DimensionlessJerk,
        SPARC,
    ]

    for fn in ext_features:
        with pytest.raises(ValueError):
            fn().compute(array([]))


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
    assert allclose(res, 0.25, atol=0.03)


def test_Skewness(get_cubic_signal):
    x = get_cubic_signal(2.0, 1.0, 1.0, 1.0, 0.0)

    res = Skewness().compute(x)

    assert allclose(res, 1.0, rtol=0.01)


def test_Kurtosis(get_cubic_signal):
    x = get_cubic_signal(2.0, 1.0, 1.0, 1.0, 0.0)

    res = Kurtosis().compute(x)

    assert allclose(res, -0.1897956, rtol=1e-5)


def test_DominantFrequency(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 5.0], 0.0)

    df_low = DominantFrequency(padlevel=2, low_cutoff=0.0, high_cutoff=2.5)
    df_high = DominantFrequency(padlevel=2, low_cutoff=2.5, high_cutoff=15.0)
    df_all = DominantFrequency(padlevel=2, low_cutoff=0.0, high_cutoff=15.0)

    res_low = df_low.compute(x, fs=fs)
    res_high = df_high.compute(x, fs=fs)
    res_all = df_all.compute(x, fs=fs)

    assert isclose(res_low, res_all)  # 1.0Hz peak has higher amplitude
    assert isclose(res_low, 1.0, atol=0.03)  # short signal, won't be exact
    assert isclose(res_high, 5.0, atol=0.03)  # short signal, won't be exact


def test_DominantFrequencyValue(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 5.0], 0.0)

    # use 1024 samples for the FFT (pad = 1 -> *2)
    df_low = DominantFrequencyValue(padlevel=1, low_cutoff=0.0, high_cutoff=2.5)
    df_high = DominantFrequencyValue(padlevel=1, low_cutoff=2.5, high_cutoff=15.0)
    df_all = DominantFrequencyValue(padlevel=1, low_cutoff=0.0, high_cutoff=15.0)

    res_low = df_low.compute(x, fs=fs)
    res_high = df_high.compute(x, fs=fs)
    res_all = df_all.compute(x, fs=fs)

    # values are somewhat close, but with padding there is some noise in the
    # transform
    assert isclose(res_low, res_high, atol=0.05)
    assert isclose(res_low, 0.5, atol=0.03)
    assert isclose(res_all, 0.4, atol=0.03)


def test_PowerSpectralSum(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 5.0], 0.0)

    df_low = PowerSpectralSum(padlevel=1, low_cutoff=0.0, high_cutoff=2.5)
    df_high = PowerSpectralSum(padlevel=1, low_cutoff=2.5, high_cutoff=15.0)
    df_all = PowerSpectralSum(padlevel=1, low_cutoff=0.0, high_cutoff=15.0)

    res_low = df_low.compute(x, fs=fs)
    res_high = df_high.compute(x, fs=fs)
    res_all = df_all.compute(x, fs=fs)

    # values are somewhat close, but with padding there is some noise in the
    # transform
    assert isclose(res_low, res_high, atol=0.01)
    # as the values should be singular spikes, almost all values lie within
    # a +-0.5 hz window, but there is noise
    assert isclose(res_low, 1.0, atol=0.03)
    assert isclose(res_all, 0.8, atol=0.03)


def test_RangePowerSum(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 0.5], 0.0)

    df_all = RangePowerSum(padlevel=1, low_cutoff=0.0, high_cutoff=15.0)
    df_low = RangePowerSum(padlevel=1, low_cutoff=0.0, high_cutoff=0.75, normalize=True)

    res_all = df_all.compute(x, fs=fs)
    res_low = df_low.compute(x, fs=fs)

    assert res_low <= 1.0  # this should be a fraction of total power so less than 1
    assert isclose(res_low, 0.24425634)
    assert isclose(res_all, 160505.75447688)


def test_SpectralFlatness(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 5.0], 0.0)

    df_low = SpectralFlatness(padlevel=0, low_cutoff=0.0, high_cutoff=2.5)
    df_high = SpectralFlatness(padlevel=0, low_cutoff=2.5, high_cutoff=15.0)
    df_all = SpectralFlatness(padlevel=0, low_cutoff=0.0, high_cutoff=15.0)

    res_low = df_low.compute(x, fs=fs)
    res_high = df_high.compute(x, fs=fs)
    res_all = df_all.compute(x, fs=fs)

    assert isclose(res_low, -22.0, atol=0.01)
    assert isclose(res_high, -21.0, atol=0.01)
    assert isclose(res_all, -25.0, atol=0.2)


def test_SpectralEntropy(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 5.0], 0.0)

    df_low = SpectralEntropy(padlevel=1, low_cutoff=0.0, high_cutoff=2.5)
    df_high = SpectralEntropy(padlevel=1, low_cutoff=2.5, high_cutoff=15.0)
    df_all = SpectralEntropy(padlevel=1, low_cutoff=0.0, high_cutoff=15.0)

    res_low = df_low.compute(x, fs=fs)
    res_high = df_high.compute(x, fs=fs)
    res_all = df_all.compute(x, fs=fs)

    assert isclose(res_low, 0.45, atol=0.01)
    assert isclose(res_high, 0.30, atol=0.01)
    assert isclose(res_all, 0.40, atol=0.02)


def test_Range(get_sin_signal):
    fs, x = get_sin_signal(1.25, 1.0, scale=0.0)

    assert isclose(Range().compute(x), 1.25 * 2)


def test_IQR(get_sin_signal):
    fs, x = get_sin_signal(1.25, 1.0, scale=0.0)

    assert isclose(IQR().compute(x), 1.739, atol=2e-4)


def test_RMS(get_sin_signal):
    fs, x = get_sin_signal(1.0, 1.0, scale=0.0)

    assert isclose(RMS().compute(x), sqrt(2) / 2, atol=1e-3)


def test_Autocorrelation(get_sin_signal):
    fs, x = get_sin_signal([1.0, 0.5], [1.0, 6.0], scale=0.0)

    res_1 = Autocorrelation(lag=1, normalize=True).compute(x)
    res_2 = Autocorrelation(lag=int(fs), normalize=True).compute(x)

    assert isclose(res_1, 1.0, atol=0.02)  # should be slightly off
    assert isclose(res_2, 1.0, atol=0.003)


def test_LinearSlope(get_cubic_signal):
    x = get_cubic_signal(0.0, 0.0, 1.375, -13.138, 0.0)

    assert isclose(LinearSlope().compute(x, 100.0), 1.375)


def test_ComplexityInvariantDistance(get_sin_signal):
    fs, x = get_sin_signal(1.0, 2.0, scale=0.0)

    res = ComplexityInvariantDistance(normalize=False).compute(x)
    truth = sqrt(sum(diff(x) ** 2))

    assert isclose(res, truth)


def test_RangeCountPercentage(get_sin_signal):
    fs, x = get_sin_signal(2.0, 0.2, 0.0)  # 1 cycle

    res = RangeCountPercentage(range_min=0.0, range_max=3.0).compute(x)

    # half the signal is above 0, but because it ends 1 sample short of 0.0
    # for second cycle, the value is 1 sample over half
    assert isclose(res, 0.502)


def test_RatioBeyondRSigma(get_sin_signal):
    fs, x = get_sin_signal(2.0, 0.2, 0.0)  # 1 cycle

    res = RatioBeyondRSigma(r=1.0).compute(x)
    truth = sum(abs(x) >= std(x, ddof=1)) / x.size

    assert isclose(res, truth)


def test_SignalEntropy(get_sin_signal):
    fs, x = get_sin_signal(1.0, 0.2, 0.0)  # 1 cycle

    res = SignalEntropy().compute(x)

    assert isclose(res, 0.256269)


def test_SampleEntropy(get_sin_signal):
    fs, x = get_sin_signal(1.0, 1.0, 0.0)

    res = SampleEntropy(m=4, r=1.1).compute(x)

    assert isclose(res, 0.01959076)


def test_PermutationEntropy(get_sin_signal):
    fs, x = get_sin_signal(1.0, 0.2, 0.0)

    res = PermutationEntropy(order=3, delay=1, normalize=True).compute(x)

    assert isclose(res, 0.40145)


def test_JerkMetric(get_sin_signal):
    fs, x = get_sin_signal(2.0, 1.0, 0.0)

    res = JerkMetric().compute(x, fs=fs)

    assert isclose(res, 0.136485)


class TestDimensionlessJerk:
    def test(self, get_sin_signal):
        fs, x = get_sin_signal(2.0, 1.0, 0.0)

        res = DimensionlessJerk(log=True, signal_type="acceleration").compute(x)
        res2 = DimensionlessJerk(log=False, signal_type="acceleration").compute(x)

        assert isclose(res, -6.19715)
        assert isclose(res2, -491.3467056034191)

    def test_signal_type_error(self):
        with pytest.raises(ValueError):
            DimensionlessJerk(signal_type="test")


def test_SPARC(get_sin_signal):
    fs, x = get_sin_signal(2.0, 0.4, 0.0)

    res = SPARC(padlevel=0, fc=10.0, amplitude_threshold=0.05).compute(x, fs=fs)

    assert isclose(res, -1.372184)

    fs, x = get_sin_signal(1.0, 1.0, 0.05)

    res2 = SPARC(padlevel=0, fc=10.0, amplitude_threshold=0.05).compute(x, fs)

    assert res2 < res


def test_DetailPower(get_sin_signal):
    fs, x = get_sin_signal([2.0, 0.5], [1.5, 5.0], 0.0)

    # default band is 1-3
    res_low = DetailPower(wavelet="coif4", freq_band=None).compute(x, fs=fs)
    res_high = DetailPower(wavelet="coif4", freq_band=[2.5, 15.0]).compute(x, fs=fs)
    res_all = DetailPower(wavelet="coif4", freq_band=[0.0001, 15.0]).compute(x, fs=fs)

    assert res_high < res_all
    assert res_low < res_all


def test_DetailPowerRatio(get_sin_signal):
    fs, x = get_sin_signal([2.0, 0.5], [1.5, 5.0], 0.0)

    # default band is 1-3
    res_low = DetailPowerRatio(wavelet="coif4", freq_band=None).compute(x, fs=fs)
    res_high = DetailPowerRatio(wavelet="coif4", freq_band=[2.5, 15.0]).compute(
        x, fs=fs
    )
    res_all = DetailPowerRatio(wavelet="coif4", freq_band=[0.0001, 15.0]).compute(
        x, fs=fs
    )

    assert isclose(res_high, res_low, atol=0.03)
    assert res_high < res_all
    assert res_low < res_all
