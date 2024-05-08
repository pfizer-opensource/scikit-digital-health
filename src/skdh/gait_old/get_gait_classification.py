"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from sys import version_info

from numpy import isclose, where, diff, insert, append, ascontiguousarray, int_
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt
import lightgbm as lgb

from skdh.utility import get_windowed_view
from skdh.utility.internal import rle
from skdh.features import Bank

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


def get_gait_classification_lgbm(gait_starts, gait_stops, accel, fs):
    """
    Get classification of windows of accelerometer data using the LightGBM classifier

    Parameters
    ----------
    gait_starts : {None, numpy.ndarray}
        Provided gait start indices.
    gait_stops : {None, numpy.ndarray}
        Provided gait stop indices.
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g"
    fs : float
        Sampling frequency for the data
    """
    if gait_starts is not None and gait_stops is not None:
        return gait_starts, gait_stops
    else:
        if not isclose(fs, 50.0) and not isclose(fs, 20.0):
            raise ValueError("fs must be either 50hz or 20hz.")
        suffix = "50hz" if fs == 50.0 else "20hz"

        wlen = int(fs * 3)  # window length, 3 seconds
        wstep = wlen  # non-overlapping windows
        thresh = 0.7  # mean + 1 stdev of best threshold for maximizing F1 score.
        # used to try to minimized false positives

        # band-pass filter
        sos = butter(1, [2 * 0.25 / fs, 2 * 5 / fs], btype="band", output="sos")
        accel_filt = ascontiguousarray(sosfiltfilt(sos, norm(accel, axis=1)))

        # window, data will already be in c-contiguous layout
        accel_w = get_windowed_view(accel_filt, wlen, wstep, ensure_c_contiguity=False)

        # get the feature bank
        feat_bank = Bank()  # data is already windowed
        feat_bank.load(_resolve_path("skdh.gait_old.model", "final_features.json"))

        # compute the features
        accel_feats = feat_bank.compute(accel_w, fs=fs, axis=1, index_axis=None)
        # output shape is (18, 99), need to transpose when passing to classifier

        # load the classification model
        lgb_file = str(
            _resolve_path(
                "skdh.gait_old.model", f"lgbm_gait_classifier_no-stairs_{suffix}.lgbm"
            )
        )
        bst = lgb.Booster(model_file=lgb_file)

        # predict
        gait_predictions = (
            bst.predict(accel_feats.T, raw_score=False) > thresh
        ).astype(int_)

        lengths, starts, vals = rle(gait_predictions)
        bout_starts = starts[vals == 1]
        bout_stops = bout_starts + lengths[vals == 1]

        # convert to actual values that match up with data
        bout_starts *= wstep
        bout_stops = bout_stops * wstep + (
            wlen - wstep
        )  # account for edges, if windows overlap

    return bout_starts.astype("int"), bout_stops.astype("int")
