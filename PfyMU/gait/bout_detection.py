"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import arange, zeros
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d
from warnings import warn
import lightgbm as lgb
from sys import version_info

if version_info >= (3, 7):
    from importlib import resources
else:
    import importlib_resources

from PfyMU.base import _BaseProcess
from PfyMU.utility import get_windowed_view
from PfyMU.features import Bank


def get_lgb_gait_classification(accel, fs):
    """
    Get classification of windows of accelerometer data

    Paramters
    ---------
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g"
    fs : float
        Sampling frequency of the data
    """
    goal_fs = 50.0  # goal fs for classifier
    wlen = int(goal_fs * 3)
    wstep = int(0.75 * wlen)
    thresh = 0.7  # mean + 1 standard deviation of best threshold for maximizing F1 score

    # down/up sample if necessary
    if fs != goal_fs:
        if fs < goal_fs:
            warn(f"fs ({fs:.2f}) is less than 50.0Hz. Upsampling to 50.0Hz")
        
        f_rs = interp1d(arange(0, accel.shape[0]) / fs, 1 / fs, accel, axis=0, bounds_error=False, fill_value='extrapolate')
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
        with resources.path('PfyMU.gait.models', 'final_features.json') as file_path:
            feat_bank.load(file_path)
    else:
        with importlib_resources.path('PfyMU.gait.models', 'final_features.json') as file_path:
            feat_bank.load(file_path)
    
    # compute the features
    accel_feats = feat_bank.compute(accel_w, fs=goal_fs, windowed=True)

    # load the model
    if version_info >= (3, 7):
        with resources.path('PfyMU.gait.models', 'lgbm_gait_classifier_no-stairs.lgbm') as file_path:
            bst = lgb.Booster(model_file=file_path)
    else:
        with importlib_resources.path('PfyMU.gait.models', 'lgbm_gait_classifier_no-stairs.lgbm') as file_path:
            bst = lgb.Booster(model_file=file_path)
    
    # predict
    gait_predictions = bst.predict(accel_feats)
    
    # expand the predictions to be per sample.
    tmp = zeros(accel.shape[0])

    c = zeros(accel.shape[0], dtype='int')  # keep track of overlap
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
    gait_pred_ps = tmp >= 0.5  # final gait predictions
    return gait_pred_ps
