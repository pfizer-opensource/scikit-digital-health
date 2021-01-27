"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from sys import version_info

from numpy import ndarray, array, where, diff, insert, append, ascontiguousarray, int_
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt
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


class DimensionMismatchError(Exception):
    pass


def get_gait_classification_lgbm(gait_pred, accel, fs):
    """
    Get classification of windows of accelerometer data using the LightGBM classifier

    Parameters
    ----------
    gait_pred : {None, numpy.ndarray, bool}
        Provided gait predictions
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g"
    fs : float
        Sampling frequency for the data
    """
    if gait_pred is not None:
        if isinstance(gait_pred, ndarray):
            if gait_pred.size != accel.shape[0]:
                raise DimensionMismatchError(
                    "Number of gait predictions (possibly downsampled) must match number of "
                    "acceleration samples")
            bout_starts = where(diff(gait_pred.astype(int_)) == 1)[0] + 1
            bout_stops = where(diff(gait_pred.astype(int_)) == -1)[0] + 1

            if gait_pred[0]:
                bout_starts = insert(bout_starts, 0, 0)
            if gait_pred[-1]:
                bout_starts = append(bout_stops, accel.shape[0])
        else:
            bout_starts, bout_stops = array([0]), array([accel.shape[0]])
    else:
        suffix = '50hz' if fs == 50.0 else '20hz'

        wlen = int(fs * 3)  # window length, 3 seconds
        wstep = wlen  # non-overlapping windows
        thresh = 0.7  # mean + 1 stdev of best threshold for maximizing F1 score.
        # used to try to minimized false positives

        # band-pass filter
        sos = butter(1, [2 * 0.25 / fs, 2 * 5 / fs], btype='band', output='sos')
        accel_filt = ascontiguousarray(sosfiltfilt(sos, norm(accel, axis=1)))

        # window, data will already be in c-contiguous layout
        accel_w = get_windowed_view(accel_filt, wlen, wstep, ensure_c_contiguity=False)

        # get the feature bank
        feat_bank = Bank()  # data is already windowed
        feat_bank.load(_resolve_path('skimu.gait.model', 'final_features.json'))

        # compute the features
        accel_feats = feat_bank.compute(accel_w, fs=fs, axis=1, index_axis=None)
        # output shape is (18, 99), need to transpose when passing to classifier

        # load the classification model
        lgb_file = str(
            _resolve_path('skimu.gait.model', f'lgbm_gait_classifier_no-stairs_{suffix}.lgbm')
        )
        bst = lgb.Booster(model_file=lgb_file)

        # predict
        gait_predictions = (bst.predict(accel_feats.T, raw_score=False) > thresh).astype(int_)

        bout_starts = where(diff(gait_predictions) == 1)[0] + 1  # account for n-1 samples in diff
        bout_stops = where(diff(gait_predictions) == -1)[0] + 1

        if gait_predictions[0]:
            bout_starts = insert(bout_starts, 0, 0)
        if gait_predictions[-1]:
            bout_stops = append(bout_stops, gait_predictions.size)

        assert bout_starts.size == bout_stops.size, "Starts and stops of bouts do not match"

        # convert to actual values that match up with data
        bout_starts *= wstep
        bout_stops = bout_stops * wstep + (wlen - wstep)  # account for edges, if windows overlap

    return bout_starts, bout_stops
