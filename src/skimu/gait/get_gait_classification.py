"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn
from sys import version_info

from numpy import arange, zeros
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d
import lightgbm as lgb

from skimu.utility import get_windowed_view
from skimu.features import Bank

if version_info >= (3, 7):
    from importlib import resources
else:
    import importlib_resources


def get_gait_classification_lgbm(accel, fs):
    """
    Get classification of windows of accelerometer data using the LightGBM classifier

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g"
    fs : float
        Sampling frequency of the data
    """
    if fs >= 50.0:
        goal_fs = 50.0  # goal fs for classifier
        suffix = '50hz'
    else:
        goal_fs = 20.0  # lower goal_fs if original frequency is less than 50Hz
        suffix = '20hz'

    wlen = int(goal_fs * 3)
    wstep = int(0.75 * wlen)
    thresh = 0.7  # mean + 1 standard deviation of best threshold for maximizing F1 score

    # down/up sample if necessary
    if fs != goal_fs:
        if fs < goal_fs:
            warn(f"fs ({fs:.2f}) is less than 50.0Hz. Upsampling to 50.0Hz")

        f_rs = interp1d(
            arange(0, accel.shape[0]) / fs, accel, kind='cubic', axis=0, bounds_error=False,
            fill_value='extrapolate'
        )
        accel_rs = f_rs(arange(0, accel.shape[0] / fs, 1 / goal_fs))
    else:
        accel_rs = accel

    # band-pass filter
    sos = butter(1, [2 * 0.25 / fs, 2 * 5 / fs], btype='band', output='sos')
    accel_rs_f = sosfiltfilt(sos, norm(accel_rs, axis=1))

    # window
    accel_w = get_windowed_view(accel_rs_f, wlen, wstep, ensure_c_contiguity=True)

    # get the feature bank
    feat_bank = Bank(window_length=None, window_step=None)  # make sure no windowing
    if version_info >= (3, 7):
        with resources.path('skimu.gait.model', 'final_features.json') as file_path:
            feat_bank.load(file_path)
    else:
        with importlib_resources.path('skimu.gait.model', 'final_features.json') as file_path:
            feat_bank.load(file_path)

    # compute the features
    accel_feats = feat_bank.compute(accel_w, fs=goal_fs, windowed=True)

    # load the model
    if version_info >= (3, 7):
        with resources.path(
                'skimu.gait.model', f'lgbm_gait_classifier_no-stairs_{suffix}.lgbm') as file_path:
            bst = lgb.Booster(model_file=str(file_path))
    else:
        with importlib_resources.path(
                'skimu.gait.model', f'lgbm_gait_classifier_no-stairs_{suffix}.lgbm') as file_path:
            bst = lgb.Booster(model_file=str(file_path))

    # predict
    gait_predictions = bst.predict(accel_feats, raw_score=False) > thresh

    # expand the predictions to be per sample, for the downsampled data
    tmp = zeros(accel_rs.shape[0])

    c = zeros(accel_rs.shape[0], dtype='int')  # keep track of overlap
    for i, p in enumerate(gait_predictions):
        i1 = i * wstep
        i2 = i1 + wlen
        # add prediction and keep track of counts effecting that cell to deal with
        # overlapping windows and therefore overlapping predictions
        tmp[i1:i2] += p
        c[i1:i2] += 1
    # make sure no divide by 0
    c[i2:] = 1

    tmp /= c
    gait_pred_sample_rs = tmp >= 0.5  # final gait predictions, ps = per sample

    # upsample the gait predictions
    f = interp1d(arange(0, accel.shape[0] / fs, 1 / goal_fs), gait_pred_sample_rs,
                 kind='previous', bounds_error=False, fill_value=0)
    gait_pred_sample = f(arange(0, accel.shape[0] / fs, 1 / fs))

    return gait_pred_sample