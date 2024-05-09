from importlib import resources

import numpy as np
from scipy.signal import iirfilter, filtfilt

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility import moving_mean, moving_sd


def _resolve_path(mod, file):
    return resources.files(mod) / file


class MotionDetectionAlgorithm(BaseProcess):
    """
    Detect periods of motion from an accelerometer signal. Threshold approach on computed rolling coefficient
    of variation (CoV).

    Input requirements:

    1. Accelerometer data is expected to be tri-axial. Orientation does not affect
    algorithm performance.

    2. Acceleration units are expected to be in G's.

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
    def predict(self, accel, fs, **kwargs):
        """
        predict(accel, fs)

        Function to detect periods of motion from an accelerometer signal.

        Parameters
        ----------
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g', collected at 20hz.
        fs : float
            Sampling frequency.

        Returns
        -------
        results : dict
            Results dictionary including detected motion (1s rolling) and raw values of rolling CoV.

        """
        # Vector Magnitude
        vmag = np.linalg.norm(accel, axis=1)

        # Low-pass filter the accelerometer vector magnitude signal to remove high frequency components
        wn = self.filter_cutoff * 2 / fs
        [b, a] = iirfilter(self.filter_cutoff, wn, btype="lowpass", ftype="butter")
        vmag_filt = filtfilt(b, a, vmag)

        # Calculate the 1s rolling coefficient of variation
        rolling_mean = moving_mean(a=vmag_filt, w_len=int(fs), skip=1)
        rolling_std = moving_sd(a=vmag_filt, w_len=int(fs), skip=1)
        rolling_cov = rolling_std / rolling_mean

        # Detect CoV values about given movement threshold
        movement = rolling_cov > self.cov_threshold

        # compile results
        results = {"movement_detected": movement, "rolling_1s_cov": rolling_cov}

        return results
