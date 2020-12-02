"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from sys import version_info

from numpy import arange, zeros, ndarray, array, interp, where, diff, insert, append, around, int_
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

    # rel_time = timestamps - timestamps[0]
    if gait_pred is None:
        if 1 / dt >= (50.0 * 0.985):  # allow for 1.5% margin on the frequency
            goal_fs = 50.0  # goal fs for classifier
            suffix = '50hz'
        else:
            goal_fs = 20.0  # lower goal_fs if original frequency is less than 50Hz
            suffix = '20hz'

        wlen = int(goal_fs * 3)
        wstep = wlen
        thresh = 0.7  # mean + 1 standard deviation of best threshold for maximizing F1 score

        # down-sample if necessary. Use +- 1.5% goal fs to account for slight sampling irregularities
        if not ((0.985 * goal_fs) < (1 / dt) < (1.015 * goal_fs)):
            """
            Using numpy's interp function here because it is a lot more memory efficient, while
            achieving the same results as interp1d(kind='linear'). Cubic interpolation using 
            scipy is a massive memory hog (goes from 1.5k Mb to 7k Mb for a 7-14 day file)
            """
            _t = arange(timestamps[0], timestamps[-1], 1 / goal_fs)
            accel_rs = zeros((_t.size, 3))
            for i in range(3):
                accel_rs[:, i] = interp(_t, timestamps, accel[:, i])
        else:
            accel_rs = accel

        # band-pass filter
        sos = butter(1, [2 * 0.25 / goal_fs, 2 * 5 / goal_fs], btype='band', output='sos')
        accel_rs = sosfiltfilt(sos, norm(accel_rs, axis=1))

        # window
        accel_w = get_windowed_view(accel_rs, wlen, wstep, ensure_c_contiguity=True)

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
        gait_predictions = (bst.predict(accel_feats, raw_score=False) > thresh).astype(int_)

        """
        Re-doing gait prediction upsampling/return, should be able to fully remove the 
        get_gait_bouts_function
        """
        bout_starts = where(diff(gait_predictions) == 1)[0] + 1  # account for n-1 samples
        bout_stops = where(diff(gait_predictions) == -1)[0] + 1

        if gait_predictions[0]:
            bout_starts = insert(bout_starts, 0, 0)
        if gait_predictions[-1]:
            bout_stops = append(bout_stops, gait_predictions.size)

        assert bout_starts.size == bout_stops.size, "Starts and stops of bouts do not match"

        # convert to per sample values, and upsample to original frequency
        bout_starts *= wstep
        bout_stops = bout_stops * wstep + (wlen - wstep)  # accounts for edges, or if windows overlap

        bout_starts = around(bout_starts / (dt * goal_fs), decimals=0).astype(int_)
        bout_stops = around(bout_stops / (dt * goal_fs), decimals=0).astype(int_)

        if gait_predictions[0]:
            bout_starts = insert(bout_starts, 0, 0)
        if gait_predictions[-1]:
            bout_stops = append(bout_stops, accel.shape[0])
    else:
        if isinstance(gait_pred, ndarray):
            if gait_pred.size != accel.shape[0]:
                raise ValueError('Number of gait predictions must much number of accel samples')
            bout_starts = where(diff(gait_pred.astype(int_)) == 1)[0] + 1
            bout_stops = where(diff(gait_pred.astype(int_)) == -1)[0] + 1

            if gait_pred[0]:
                bout_starts = insert(bout_starts, 0, 0)
            if gait_pred[-1]:
                bout_stops = append(bout_stops, accel.shape[0])
        else:
            bout_starts = array([0])
            bout_stops = array([accel.shape[0]])

    return bout_starts, bout_stops
