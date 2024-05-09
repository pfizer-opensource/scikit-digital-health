"""
Motion detection for triaxial accelerometer data.

Yiorgos Christakis
Copyright (c) 2024, Pfizer Inc. All rights reserved.
"""

from warnings import warn
from importlib import resources

import numpy as np
from numpy import round
from scipy.signal import iirfilter, filtfilt, butter, sosfiltfilt

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility import moving_mean, moving_sd
from skdh.utility.internal import apply_resample


def _resolve_path(mod, file):
    return resources.files(mod) / file


class MotionDetectMahadevanEtAl(BaseProcess):
    """
    Detect periods of motion from an accelerometer signal. Threshold approach on computed 1s rolling coefficient
    of variation (CoV).

    Method implemented as described in:
    Mahadevan, N., Christakis, Y., Di, J. et al. Development of digital measures for nighttime scratch and sleep using
    wrist-worn wearable devices. npj Digit. Med. 4, 42 (2021). https://doi.org/10.1038/s41746-021-00402-x

    Input requirements:

    1. Accelerometer data must be collected with a sampling frequency of at least
    20hz.

    2. Accelerometer data is expected to be tri-axial. Orientation does not affect
    algorithm performance.

    3. Acceleration units are expected to be in G's.

    4. A minimum of 20 samples (or the equivalent of a single 1-second window at 20hz) is
    required for predictions to be made.

    Parameters
    ----------
    filter_cutoff : int
        Low pass filter cutoff. Data filtered prior to CoV calculation.
    cov_threshold : float
        Threshold computed rolling CoV to determine motion present.

    """

    def __init__(self, filter_cutoff=6, cov_threshold=0.022558169349678834):
        super().__init__(filter_cutoff=filter_cutoff, cov_threshold=cov_threshold)
        self.filter_cutoff = filter_cutoff
        self.cov_threshold = cov_threshold

    @handle_process_returns(results_to_kwargs=False)
    def predict(self, time, accel, fs=None, **kwargs):
        """
        predict(time, accel, fs)

        Function to detect periods of motion from an accelerometer signal.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g'.
        fs : float
            Sampling rate. Default None. If not provided, will be inferred.

        Returns
        -------
        results : dict
            Results dictionary including detected motion (1s rolling) and raw values of 1s rolling CoV.

        """
        # check input requirements are met
        time, accel, fs = self._check_input(time, accel)

        # Vector Magnitude
        vmag = np.linalg.norm(accel, axis=1)

        # Low-pass filter the accelerometer vector magnitude signal to remove high frequency components
        # Note: iirfilter with filtfilt used to match legacy implementation
        wn = self.filter_cutoff * 2 / fs
        [b, a] = iirfilter(self.filter_cutoff, wn, btype="lowpass", ftype="butter")
        vmag_filt = filtfilt(b, a, vmag)

        # Calculate the 1s rolling coefficient of variation
        # Note: added a non-zero term in denominator to prevent divide by zero errors
        rolling_std, rolling_mean = moving_sd(a=vmag_filt, w_len=int(fs), skip=1)
        rolling_cov = rolling_std / (rolling_mean + 1e-12)

        # Detect CoV values above movement threshold
        movement = rolling_cov > self.cov_threshold

        # Group results
        results = {"movement_detected": movement, "rolling_1s_cov": rolling_cov}

        return results

    @staticmethod
    def _check_input(time, accel, fs=None):
        """
        Checks that input meets requirements (see class docstring). Infers fs if necessary.

        Parameters
        ----------
        time : array-like
            Numpy array of unix timestamps. Units of seconds.
        accel : array-like
            Numpy array of triaxial accelerometer data.
        fs : float
            Sampling rate. Default None. If not provided, will be inferred.

        Returns
        -------
        time_ds : array-like
        accel_ds : array-like
        fs : float
            Sampling frequency.

        """
        # check # of columns
        _, c = accel.shape
        if not (c == 3):
            raise ValueError("Input expected to have 3 columns, but found " + str(c))

        # units must be in G's (mean of magnitude of x,y,z across the entire signal < 4)
        avg = np.mean(np.linalg.norm(accel, axis=1))
        if not (avg < 4):
            raise ValueError(
                "Input expected to have units of G's, but mean signal magnitude greater than 4."
            )

        # check fs
        fs = round(1 / np.mean(np.diff(time)), 3) if fs is None else fs
        if fs < 20.0:
            raise ValueError(f"Frequency ({fs:.2f}Hz) is too low (<20Hz).")

        # check
        r, _ = accel.shape
        if not (r >= 20):
            raise ValueError(
                "Input at 20hz expected to have at least 20 rows, but found " + str(r)
            )

        return time, accel, fs


class MotionDetectJiEtAl(BaseProcess):
    """
    Detect periods of motion from a 20hz accelerometer signal. Threshold approach on computed 1s rolling coefficient
    of variation (CoV) and max standard deviation across three axes.

    Method implemented as described in:
    Ji, J., Venderley, J., Zhang, H. et al. Assessing nocturnal scratch with actigraphy in atopic dermatitis
    patients. npj Digit. Med. 6, 72 (2023). https://doi.org/10.1038/s41746-023-00821-y

    Input requirements:

    1. Accelerometer data is expected to be tri-axial. Orientation does not affect
    algorithm performance.

    2. Acceleration units are expected to be in G's.

    3. Accelerometer data is expected to be sampled at exactly 20hz.

    4. A minimum of 20 samples (or the equivalent of a single 1-second window) is
    required for predictions to be made.

    Parameters
    ----------
    lp_filter_cutoff : float
        Low pass filter cutoff. Data filtered prior to CoV calculation.
    lp_filter_order : int
        Low pass filter order. Data filtered prior to CoV calculation.
    hp_filter_cutoff : float
        High pass filter cutoff. Data filtered prior to CoV calculation.
    hp_filter_order : int
        High pass filter order. Data filtered prior to CoV calculation.
    cov_threshold : float
        Threshold computed rolling CoV to determine motion present.
    sd_threshold : float
        Threshold for max standard deviation to determine motion present.

    """

    def __init__(
        self,
        lp_filter_cutoff=3.0,
        lp_filter_order=6,
        hp_filter_cutoff=0.25,
        hp_filter_order=1,
        cov_threshold=0.41,
        sd_threshold=0.013,
    ):
        super().__init__(
            lp_filter_cutoff=lp_filter_cutoff,
            lp_filter_order=lp_filter_order,
            hp_filter_cutoff=hp_filter_cutoff,
            hp_filter_order=hp_filter_order,
            cov_threshold=cov_threshold,
            sd_threshold=sd_threshold,
        )
        self.lp_filter_cutoff = lp_filter_cutoff
        self.lp_filter_order = lp_filter_order
        self.hp_filter_cutoff = hp_filter_cutoff
        self.hp_filter_order = hp_filter_order
        self.cov_threshold = cov_threshold
        self.sd_threshold = sd_threshold

    @handle_process_returns(results_to_kwargs=False)
    def predict(self, time, accel, fs=None, **kwargs):
        """
        predict(time, accel, fs)

        Function to detect periods of motion from an accelerometer signal.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g', collected at 20hz.
        fs : float
            Sampling rate. Default None. If not provided, will be inferred.

        Returns
        -------
        results : dict
            Results dictionary with motion predictions (1hz boolean; true indicates motion) and associated timestamps.

        """
        # check input requirements are met
        time, accel, fs = self._check_input(time, accel, fs)

        # Branch 1: Rolling CoV thresholding
        # 1. Vector Magnitude
        vmag = np.linalg.norm(accel, axis=1)

        # 2. Low-pass filter the accelerometer vector magnitude signal to remove high frequency components
        sos_lp = butter(
            N=self.lp_filter_order,
            Wn=2 * self.lp_filter_cutoff / fs,
            btype="lowpass",
            output="sos",
        )
        vmag_lp = sosfiltfilt(sos_lp, vmag, axis=0)

        # 3. High-pass filter the accelerometer vector magnitude signal to remove DC component
        sos_hp = butter(
            N=self.hp_filter_order,
            Wn=2 * self.hp_filter_cutoff / fs,
            btype="highpass",
            output="sos",
        )
        vmag_lp_hp = sosfiltfilt(sos_hp, vmag_lp, axis=0)

        # 4. Calculate the 1s rolling coefficient of variation
        rolling_std, rolling_mean = moving_sd(a=vmag_lp_hp, w_len=int(fs), skip=1)
        rolling_cov = rolling_std / (np.abs(rolling_mean) + 0.01)

        # 5. Detect CoV values greater than movement threshold
        movement = rolling_cov >= self.cov_threshold

        # 6. Get non-overlapping mean of predictions. If average of predictions in window is >=0.5 (at least half of
        # the CoVs are greater than the threshold), then movement is true.
        windowed_movement = moving_mean(a=movement, w_len=int(fs), skip=int(fs))
        cov_predictions = windowed_movement >= 0.5

        # Branch 2: Max standard deviation thresholding
        # 1. Moving SD with no overlap
        sd, _ = moving_sd(a=accel, w_len=int(fs), skip=int(fs), axis=0)

        # 2. Max in each second
        maxsd = np.max(sd, axis=1)

        # 3. Threshold maxsd
        maxsd_predictions = maxsd >= self.sd_threshold

        # Final step: OR of branch 1 and 2
        predictions = cov_predictions | maxsd_predictions[0:len(cov_predictions)]

        # Sample time array to 1s resolution
        time_1s = time[0 :: int(fs)].copy()[0:len(predictions)]

        # compile results
        results = {"movement_detected": predictions, "movement_time": time_1s}

        return results

    @staticmethod
    def _check_input(time, accel, fs=None):
        """
        Checks that input meets requirements (see class docstring). Downsamples data >20hz to 20hz.

        Parameters
        ----------
        time : array-like
            Numpy array of unix timestamps. Units of seconds.
        accel : array-like
            Numpy array of triaxial accelerometer data.
        fs : float
            Sampling rate. Default None. If not provided, will be inferred.

        Returns
        -------
        time_ds : array-like
        accel_ds : array-like
        fs : float
            Sampling frequency.

        """
        # check # of columns
        _, c = accel.shape
        if not (c == 3):
            raise ValueError("Input expected to have 3 columns, but found " + str(c))

        # units must be in G's (mean of magnitude of x,y,z across the entire signal < 4)
        avg = np.mean(np.linalg.norm(accel, axis=1))
        if not (avg < 4):
            raise ValueError(
                "Input expected to have units of G's, but mean signal magnitude greater than 4."
            )

        # check fs & downsample if necessary
        fs = round(1 / np.mean(np.diff(time)), 3) if fs is None else fs
        if fs < 20.0:
            raise ValueError(f"Frequency ({fs:.2f}Hz) is too low (<20Hz).")
        elif fs > 20.0:
            warn(
                "Frequency is > 20Hz. Downsampling to 20Hz.",
                UserWarning,
            )
            time_ds, (accel_ds,) = apply_resample(
                time=time,
                goal_fs=20.0,
                data=(accel,),
                fs=fs,
            )
        else:
            time_ds = time
            accel_ds = accel

        # check
        r, _ = accel_ds.shape
        if not (r >= 60):
            raise ValueError(
                "Input at 20hz expected to have at least 60 rows, but found " + str(r)
            )

        return time_ds, accel_ds, fs
