"""
Gait bout acceleration pre-processing functions.

Lukas Adamowicz
Copyright (c) 2023, Pfizer Inc. All rights reserved
"""

from numpy import (
    mean,
    median,
    argmax,
    sign,
    abs,
    diff,
    array,
    logspace,
    log,
    exp,
    sum,
    max as npmax,
)
from scipy.signal import detrend, butter, sosfiltfilt, find_peaks, correlate
import pywt

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility import (
    correct_accelerometer_orientation,
    compute_window_samples,
    get_windowed_view,
)
from skdh.gait.gait_metrics import gait_metrics


class PreprocessGaitBout(BaseProcess):
    """
    Preprocess acceleration data for gait using the newer/V2 method.

    Parameters
    ----------
    correct_orientation : bool, optional
        Correct the accelerometer orientation if it is slightly mis-aligned
        with the anatomical axes. Default is True.
    filter_cutoff : float, optional
        Low-pass filter cutoff in Hz. Default is 20.0
    filter_order : int, optional
        Low-pass filter order. Default is 4.
    ap_axis_filter_kw : {None, dict}, optional
        Key-word arguments for the filter applied to the acceleration data before
        cross-correlation when estimating the AP axis.
        If None (default), the following are used:

        - `N`: 4
        - `Wn`: [0.25, 7.5] - NOTE, this should be in Hz, not radians.
          fs will be passed into the filter setup at filter creation time.
        - `btype`: band
        - `output`: sos - NOTE that this will always be set/overriden

        See :func:`scipy.signal.butter` for full options.
    """

    def __init__(
        self,
        correct_orientation=True,
        filter_cutoff=20.0,
        filter_order=4,
        ap_axis_filter_kw=None,
    ):
        super().__init__(
            correct_orientation=correct_orientation,
            filter_cutoff=filter_cutoff,
            filter_order=filter_order,
        )

        self.corr_orient = correct_orientation
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order

        if ap_axis_filter_kw is None:
            ap_axis_filter_kw = {
                "N": 4,
                "Wn": array([0.25, 7.5]),
                "btype": "band",
            }

        ap_axis_filter_kw.update({"output": "sos"})  # ALWAYS set this
        self.ap_axis_filter_kw = ap_axis_filter_kw

    def get_ap_axis(self, fs, accel, v_axis):
        """
        Estimate the AP axis index

        Parameters
        ----------
        fs
        accel
        v_axis

        Returns
        -------
        ap_axis : {0, 1, 2}
            Index of the AP axis in the acceleration data
        """
        # filter acceleration
        sos = butter(fs=fs, **self.ap_axis_filter_kw)
        accel_filt = sosfiltfilt(sos, accel, axis=0)
        axes = {0, 1, 2}
        # drop v-axis
        a1, a2 = axes.difference([v_axis])

        c_v1 = correlate(accel_filt[:, v_axis], accel_filt[:, a1])
        c_v2 = correlate(accel_filt[:, v_axis], accel_filt[:, a2])

        idx = argmax([npmax(abs(c_v1)), npmax(abs(c_v2))])

        return [a1, a2][idx]  # index into the remaining axes

    @staticmethod
    def get_ap_axis_sign(fs, accel, ap_axis):
        """
        Estimate the sign of the AP axis

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        accel : numpy.ndarray
        ap_axis : int
            Anterior-Posterior axis

        Returns
        -------
        ap_axis_sign : {-1, 1}
            Sign of the AP axis.
        """
        sos = butter(4, [2 * 0.25 / fs, 2 * 7.0 / fs], output="sos", btype="band")
        ap_acc_f = sosfiltfilt(sos, accel[:, ap_axis])

        mx, mx_meta = find_peaks(ap_acc_f, prominence=0.05)
        med_prom = median(mx_meta["prominences"])
        mask = mx_meta["prominences"] > (0.75 * med_prom)

        left_med = median(mx[mask] - mx_meta["left_bases"][mask])
        right_med = median(mx_meta["right_bases"][mask] - mx[mask])

        sign = -1 if (left_med < right_med) else 1

        return sign

    @staticmethod
    def get_step_time(fs, accel, ap_axis):
        """
        Estimate the average step time for the walking bout.

        Parameters
        ----------
        fs
        accel
        ap_axis

        Returns
        -------
        mean_step_time : float
            Mean step time for the walking bout
        """
        # span a range of scales
        scale1 = pywt.frequency2scale("gaus1", 0.5 / fs)
        scale2 = pywt.frequency2scale("gaus1", 5.0 / fs)
        scales = logspace(log(scale1), log(scale2), 10, base=exp(1))

        coefs, _ = pywt.cwt(accel[:, ap_axis], scales, "gaus1")
        csum = sum(coefs, axis=0)  # "power" in that frequency band

        # window - 5 second windows, 50% overlap
        samp, step = compute_window_samples(fs, 5.0, 0.5)
        coefsum_w = get_windowed_view(csum, samp, step, ensure_c_contiguity=True)

        # auto covariance
        ac_w = gait_metrics._autocovariancefn(coefsum_w, samp - 10, biased=True, axis=1)

        first_peaks = []
        for i in range(ac_w.shape[0]):
            pks, _ = find_peaks(ac_w[i, :], height=0.0)
            try:
                first_peaks.append(pks[0])
            except IndexError:
                continue

        if len(first_peaks) < (0.25 * ac_w.shape[0]):
            raise ValueError(
                "Not enough valid autocovariance windows to estimate step frequency."
            )
        step_samples = median(first_peaks)

        return step_samples / fs

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, *, time, accel, fs=None, v_axis=None, ap_axis=None, **kwargs):
        """
        predict(time, accel, *, fs=None, v_axis=None, ap_axis=None)

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
        v_axis : {None, 0, 1, 2}, optional
            Index of the vertical axis in the acceleration data. Default is None.
            If None, will be estimated from the acceleration data.
        ap_axis : {None, 0, 1, 2}, optional
            Index of the Anterior-Posterior axis in the acceleration data.
            Default is None. If None, will be estimated from the acceleration data.

        Returns
        -------
        results : dict
            Dictionary with the following items that can be used for future
            processing steps:

            - `v_axis`: provided or estimated vertical axis index.
            - `v_axis_est`: estimated vertical axis index.
            - `v_axis_sign`: sign of the vertical axis.
            - `ap_axis`: provided or estimated AP axis index.
            - `ap_axis_est`: estimated AP axis index.
            - `ap_axis_sign`: estimated sign of the AP axis.
            - `mean_step_freq`: estimated mean step frequency during this gait bout.
            - `accel_filt`: filtered and detrended acceleration for this gait bout.
        """
        # calculate fs if we need to
        fs = 1 / mean(diff(time)) if fs is None else fs

        # estimate accelerometer axes if necessary
        acc_mean = mean(accel, axis=0)
        v_axis_est = argmax(abs(acc_mean))  # always estimate for testing purposes
        if v_axis is None:
            v_axis = v_axis_est

        # always compute the sign
        v_axis_sign = sign(acc_mean[v_axis])

        # always estimate for testing purposes
        ap_axis_est = self.get_ap_axis(fs, accel, v_axis)

        if ap_axis is None:
            ap_axis = ap_axis_est

        # always compute the sign
        ap_axis_sign = self.get_ap_axis_sign(fs, accel, ap_axis)

        if self.corr_orient:
            accel = correct_accelerometer_orientation(
                accel, v_axis=v_axis, ap_axis=ap_axis
            )

        # filter if possible
        if fs > (2 * self.filter_cutoff):
            sos = butter(
                self.filter_order,
                2 * self.filter_cutoff / fs,
                output="sos",
                btype="low",
            )
            accel_filt = sosfiltfilt(sos, accel, axis=0)
        else:
            accel_filt = accel

        # detrend
        accel_filt = detrend(accel_filt, axis=0)

        # estimate step frequency
        step_time = self.get_step_time(fs, accel, ap_axis)
        mean_step_freq = 1 / step_time
        # constrain the step frequency
        mean_step_freq = max(min(mean_step_freq, 5.0), 0.4)

        res = {
            "v_axis": v_axis,
            "v_axis_est": v_axis_est,
            "v_axis_sign": v_axis_sign,
            "ap_axis": ap_axis,
            "ap_axis_est": ap_axis_est,
            "ap_axis_sign": ap_axis_sign,
            "mean_step_freq": mean_step_freq,
            "accel_filt": accel_filt,
        }

        return res
