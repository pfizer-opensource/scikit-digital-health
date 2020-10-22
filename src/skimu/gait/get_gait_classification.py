"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from sys import version_info

from numpy import arange, zeros, ndarray, full, bool_
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d
import lightgbm as lgb

from skimu.utility import get_windowed_view
from skimu.features import Bank

if version_info >= (3, 7):
    from importlib import resources
else:  # pragma: no cover
    import importlib_resources


def _resolve_path(mod, file):
    if version_info >= (3, 7):
        with resources.path(mod, file) as file_path:
            path = file_path
    else:  # pragma: no cover
        with importlib_resources.path(mod, file) as file_path:
            path = file_path

    return path


def get_gait_classification_lgbm(gait_pred, accel, dt, timestamps):
    """
    Get classification of windows of accelerometer data using the LightGBM classifier

    Parameters
    ----------
    gait_pred : {None, numpy.ndarray, bool}
        Provided gait predictions
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g"
    dt : float
        Sampling period for the data
    timestamps : numpy.ndarray
        Array of timestmaps (in seconds) corresponding to acceleration sampling times.
    """
    assert accel.shape[0] == timestamps.size, "`vert_accel` and `timestamps` # samples must match"

    rel_time = timestamps - timestamps[0]
    if gait_pred is None:
        if 1 / dt >= 50.0:
            goal_fs = 50.0  # goal fs for classifier
            suffix = '50hz'
        else:
            goal_fs = 20.0  # lower goal_fs if original frequency is less than 50Hz
            suffix = '20hz'

        wlen = int(goal_fs * 3)
        wstep = wlen
        thresh = 0.7  # mean + 1 standard deviation of best threshold for maximizing F1 score

        # down-sample if necessary
        if 1 / dt != goal_fs:
            f_rs = interp1d(
                rel_time, accel, kind='cubic', axis=0, bounds_error=False,
                fill_value='extrapolate'
            )
            accel_rs = f_rs(arange(0, rel_time[-1], 1 / goal_fs))
        else:
            accel_rs = accel

        # band-pass filter
        sos = butter(1, [2 * 0.25 * dt, 2 * 5 * dt], btype='band', output='sos')
        accel_rs_f = sosfiltfilt(sos, norm(accel_rs, axis=1))

        # window
        accel_w = get_windowed_view(accel_rs_f, wlen, wstep, ensure_c_contiguity=True)

        # get the feature bank
        feat_bank = Bank(window_length=None, window_step=None)  # make sure no windowing
        feat_bank.load(_resolve_path('skimu.gait.model', 'final_features.json'))

        # compute the features
        accel_feats = feat_bank.compute(accel_w, fs=goal_fs, windowed=True)

        # load the model
        lgb_file = str(
            _resolve_path('skimu.gait.model', f'lgbm_gait_classifier_no-stairs_{suffix}.lgbm')
        )
        bst = lgb.Booster(model_file=lgb_file)

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
        f = interp1d(
            arange(0, rel_time[-1] + 1 / goal_fs, 1 / goal_fs)[:gait_pred_sample_rs.size],
            gait_pred_sample_rs,
            kind='previous', bounds_error=False, fill_value=0
        )
        gait_pred_sample = f(rel_time)
    else:
        if isinstance(gait_pred, ndarray):
            if gait_pred.size != accel.shape[0]:
                raise ValueError('Number of gait predictions must much number of accel samples')
            gait_pred_sample = gait_pred
        else:
            gait_pred_sample = full(accel.shape[0], True, dtype=bool_)

    return gait_pred_sample
