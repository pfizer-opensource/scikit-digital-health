import pytest
import numpy as np
from lightgbm import Booster
from skdh.context import Ambulation


# Test input check on real data
def test_input_check(ambulation_positive_data):
    time, accel = ambulation_positive_data
    amb = Ambulation()
    assert len(amb._check_input(time, accel)) == 3


# Test preprocessing on real data
def test_preprocessing(ambulation_positive_data):
    time, accel = ambulation_positive_data
    r, c = accel.shape
    win_len = 60

    ppr = Ambulation()._preprocess(accel)
    assert ppr.shape == (r // win_len, win_len)


# Test feature extraction on real data
def test_feature_extraction(ambulation_positive_data):
    time, accel = ambulation_positive_data
    r, c = accel.shape
    win_len = 60

    ppr = Ambulation()._preprocess(accel)
    true_names = [
        "MAG_Mean()",
        "MAG_MeanCrossRate()",
        "MAG_StdDev()",
        "MAG_Skewness()",
        "MAG_Kurtosis()",
        "MAG_Range()",
        "MAG_IQR()",
        "MAG_Autocorrelation(lag=1, normalize=True)",
        "MAG_Autocorrelation(lag=5, normalize=True)",
        "MAG_Autocorrelation(lag=10, normalize=True)",
        "MAG_Autocorrelation(lag=20, normalize=True)",
        "MAG_LinearSlope()",
        "MAG_SignalEntropy()",
        "MAG_SampleEntropy(m=4, r=1.0)",
        "MAG_PermutationEntropy(order=3, delay=1, normalize=False)",
        "MAG_RangeCountPercentage(range_min=0, range_max=1.0)",
        "MAG_JerkMetric()",
        "MAG_SPARC(padlevel=4, fc=10.0, amplitude_threshold=0.05)",
        "MAG_DominantFrequency(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)",
        "MAG_DominantFrequencyValue(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)",
        "MAG_PowerSpectralSum(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)",
        "MAG_SpectralFlatness(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)",
        "MAG_SpectralEntropy(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)",
        "MAG_DetailPower(wavelet='coif4', freq_band=[1.0, 8.0])",
        "MAG_DetailPowerRatio(wavelet='coif4', freq_band=[1.0, 8.0])",
    ]

    ftr, names = Ambulation()._compute_features(ppr)
    assert (ftr.shape == (r // win_len, len(true_names))) & (names == true_names)


# Test model loading
def test_model_loading():
    mdl = Ambulation()._load_model("ambulation_model.txt")
    print(type(mdl))
    assert isinstance(mdl, Booster)


# Test input check row requirement
def test_input_check_rows():
    with pytest.raises(ValueError) as e_info:
        Ambulation()._check_input(
            time=np.arange(0, 59 / 20, 1 / 20), accel=np.ones([59, 3])
        )


# Test input check column requirement
def test_input_check_columns():
    with pytest.raises(ValueError) as e_info:
        Ambulation()._check_input(
            time=np.arange(0, 60 / 20, 1 / 20), accel=np.ones([60, 2])
        )


# Test input check sample rate requirement
def test_input_check_fs():
    with pytest.raises(ValueError) as e_info:
        Ambulation()._check_input(
            time=np.arange(0, 120 / 10, 1 / 10), accel=np.ones([120, 3])
        )


# Integration test 1: predict method on ambulation data
def test_integration_ambulation(ambulation_positive_data):
    time, accel = ambulation_positive_data
    res = Ambulation().predict(time=time, accel=accel)
    prd = res["ambulation_3s_epochs_predictions"]
    assert (
        sum(prd) > len(prd) * 0.85
    )  # confirm detecting more than 85% of walking data as ambulation


# Integration test 2: predict method on non-ambulation data
def test_integration_non_ambulation(ambulation_negative_data):
    time, accel = ambulation_negative_data
    res = Ambulation().predict(time=time, accel=accel)
    prd = res["ambulation_3s_epochs_predictions"]
    assert (
        sum(prd) < len(prd) * 0.15
    )  # confirm detecting less than 15% of flat data as ambulation


# Integration test 3: predict method on non ambulation data requiring downsampling
def test_integration_ambulation_downsample(ambulation_negative_data_50hz):
    time, accel = ambulation_negative_data_50hz
    res = Ambulation().predict(time=time, accel=accel)
    prd = res["ambulation_3s_epochs_predictions"]
    assert (
        sum(prd) < len(prd) * 0.15
    )  # confirm detecting less than 15% of walking data as ambulation on downsampled data
