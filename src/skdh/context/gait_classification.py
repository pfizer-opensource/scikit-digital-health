"""
Gait detection via LGBM classifier

Lukas Adamowicz
Copyright (c) 2023, Pfizer Inc. All rights reserved.
"""

from sys import version_info
from warnings import warn

from numpy import (
    mean,
    diff,
    ascontiguousarray,
    int_,
    around,
    interp,
    arange,
    concatenate,
)
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt
import lightgbm as lgb

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility.exceptions import LowFrequencyError
from skdh.utility.internal import apply_resample, rle
from skdh.utility.windowing import get_windowed_view

from skdh.features import Bank

from importlib import resources


def _resolve_path(mod, file):
    return resources.files(mod) / file


class PredictGaitLumbarLgbm(BaseProcess):
    """
    Process lumbar acceleration data to predict bouts of gait using a Light Gradient
    Boosted model.

    Predictions are computed on non-overlappping 3-second windows.

    Parameters
    ----------
    downsample_aa_filter : bool, optional
        Apply an anti-aliasing filter when downsampling accelerometer data. Default
        is True.
    """

    def __init__(self, downsample_aa_filter=True):
        super().__init__(
            downsample_aa_filter=downsample_aa_filter,
        )

        self.downsample_aa_filter = downsample_aa_filter

    @handle_process_returns(results_to_kwargs=False)
    def predict(self, *, time, accel, fs=None, tz_name=None, **kwargs):
        """
        predict(*, time, accel, fs=None)

        Predict gait bouts.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by a centrally mounted lumbar
            inertial measurement device, in units of 'g'.
        fs : float, optional
            Sampling frequency in Hz of the accelerometer data. If not provided,
            will be computed form the timestamps.

        Other Parameters
        ----------------
        tz_name : {None, str}, optional
            IANA time-zone name for the recording location if passing in `time` as
            UTC timestamps. Can be ignored if passing in naive timestamps.

        Returns
        -------
        gait_bouts : numpy.ndarray
            (N, 2) array of indices of the starts (column 1) and stops (column 2)
            of gait bouts.

        Raises
        ------
        LowFrequencyError
            If the sampling frequency is less than 20hz.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            fs=fs,
            tz_name=tz_name,
            **kwargs,
        )

        # compute fs if necessary
        fs = 1 / mean(diff(time)) if fs is None else fs
        if fs < 20.0:
            raise LowFrequencyError(f"Frequency ({fs:.2}Hz) is too low (<20Hz).")
        if fs < 50.0:
            warn(
                f"Frequency ({fs:.2}Hz) is less than 50Hz. Downsampling to 20Hz.",
                UserWarning,
            )

        # handle frequency and downsampling
        goal_fs = 50.0 if fs >= 50.0 else 20.0
        if fs != goal_fs:
            time_ds, (accel_ds,), *_ = apply_resample(
                goal_fs=goal_fs,
                time=time,
                data=(accel,),
                indices=(),
                aa_filter=self.downsample_aa_filter,
                fs=fs,
            )
        else:
            time_ds = time
            accel_ds = accel

        # do the classification
        suffix = "50hz" if goal_fs == 50.0 else "20hz"

        wlen = int(goal_fs * 3)  # window length of 3 seconds
        wstep = wlen  # non-overlapping windows
        thresh = 0.7  # mean + 1 stdev of best threshold for maximizing F1 score.
        # used to try to minimize false positives

        # band pass filter
        sos = butter(1, [2 * 0.25 / fs, 2 * 5.0 / fs], btype="band", output="sos")
        accel_filt = ascontiguousarray(sosfiltfilt(sos, norm(accel_ds, axis=1)))

        # window data, data will already be c-contiguous
        accel_w = get_windowed_view(accel_filt, wlen, wstep, ensure_c_contiguity=False)

        # get the feature bank
        feat_bank = Bank()  # data is already windowed
        feat_bank.load(_resolve_path("skdh.context.model", "gait_final_features.json"))

        # compute the features
        accel_feats = feat_bank.compute(accel_w, fs=goal_fs, axis=1, index_axis=None)
        # output shape is (18, 99), need to transpose when passing to classifier

        # load the classification model
        lgb_file = str(
            _resolve_path(
                "skdh.context.model", f"lgbm_gait_classifier_no-stairs_{suffix}.lgbm"
            )
        )
        bst = lgb.Booster(model_file=lgb_file)

        # predict
        gait_predictions = (
            bst.predict(accel_feats.T, raw_score=False) > thresh
        ).astype(int_)

        lengths, starts, vals = rle(gait_predictions)
        bout_starts_ds = starts[vals == 1]
        bout_stops_ds = bout_starts_ds + lengths[vals == 1]

        # convert to actual values that match up with DOWNSAMPLED data
        bout_starts_ds *= wstep
        # account for edges, if windows overlap
        bout_stops_ds = bout_stops_ds * wstep + (wlen - wstep)

        # make sure that we dont go over the array limits
        try:
            bout_stops_ds[-1] = min(bout_stops_ds[-1], time_ds.size - 1)
        except IndexError:
            pass

        # upsample indices back to original data
        bout_starts = around(
            interp(time_ds[bout_starts_ds], time, arange(time.size))
        ).astype(int_)
        bout_stops = around(
            interp(time_ds[bout_stops_ds], time, arange(time.size))
        ).astype(int_)

        bouts = concatenate((bout_starts, bout_stops)).reshape((2, -1)).T

        results = {
            "Gait Bout Start": time[bout_starts],
            "Gait Bout Stop": time[bout_stops],
            "Gait Bout Start Index": bout_starts,
            "Gait Bout Stop Index": bout_stops,
            "Timezone": tz_name,
        }

        return results, {"gait_bouts": bouts}
