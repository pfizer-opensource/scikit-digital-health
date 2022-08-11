"""
Function for getting gait events from an accelerometer signal

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import fft, argmax, std, abs, argsort, corrcoef, mean, sign
from scipy.signal import detrend, butter, sosfiltfilt, find_peaks
from scipy.integrate import cumtrapz
from pywt import cwt

from skdh.utility import correct_accelerometer_orientation
from skdh.gait.gait_endpoints import gait_endpoints


def get_cwt_scales(use_optimal_scale, vertical_velocity, original_scale, fs):
    """
    Get the CWT scales for the IC and FC events.

    Parameters
    ----------
    use_optimal_scale : bool
        Use the optimal scale based on step frequency.
    vertical_velocity : numpy.ndarray
        Vertical velocity, in units of "g".
    original_scale : int
        The original/default scale for the CWT.
    fs : float
        Sampling frequency, in Hz.

    Returns
    -------
    scale1 : int
        First scale for the CWT. For initial contact events.
    scale2 : int
        Second scale for the CWT. For final contact events.
    """
    if use_optimal_scale:
        coef_scale_original, _ = cwt(vertical_velocity, original_scale, "gaus1")

        F = abs(fft.rfft(coef_scale_original[0]))
        # compute an estimate of the step frequency
        step_freq = argmax(F) / vertical_velocity.size * fs

        # IC scale: -10 * sf + 56
        # FC scale: -52 * sf + 131
        # TODO verify the FC scale equation. This it not in the paper but is a
        #  guess from the graph
        # original fs  was 250hz, hence the conversion
        scale1 = min(max(round((-10 * step_freq + 56) * (fs / 250)), 1), 90)
        scale2 = min(max(round((-52 * step_freq + 131) * (fs / 250)), 1), 90)
        # scale range is between 1 and 90
    else:
        scale1 = scale2 = original_scale

    return scale1, scale2


def get_gait_events(
    accel,
    fs,
    ts,
    orig_scale,
    filter_order,
    filter_cutoff,
    corr_accel_orient,
    use_optimal_scale,
):
    """
    Get the bouts of gait from the acceleration during a gait bout

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration during the gait bout.
    fs : float
        Sampling frequency for the acceleration.
    ts : numpy.ndarray
        Array of timestmaps (in seconds) corresponding to acceleration sampling times.
    orig_scale : int
        Original scale for the CWT.
    filter_order : int
        Low-pass filter order.
    filter_cutoff : float
        Low-pass filter cutoff in Hz.
    corr_accel_orient : bool
        Correct the accelerometer orientation.
    use_optimal_scale : bool
        Use the optimal scale based on step frequency.

    Returns
    -------
    init_contact : numpy.ndarray
        Indices of initial contacts
    final_contact : numpy.ndarray
        Indices of final contacts
    vert_accel : numpy.ndarray
        Filtered vertical acceleration
    v_axis : int
        The axis corresponding to the vertical acceleration
    """
    assert accel.shape[0] == ts.size, "`vert_accel` and `ts` size must match"

    # figure out vertical axis on a per-bout basis
    acc_mean = mean(accel, axis=0)
    v_axis = argmax(abs(acc_mean))
    va_sign = sign(acc_mean[v_axis])  # sign of the vertical acceleration

    # correct acceleration orientation if set
    if corr_accel_orient:
        # determine AP axis
        ac = gait_endpoints._autocovariancefn(
            accel, min(accel.shape[0] - 1, 1000), biased=True, axis=0
        )
        ap_axis = argsort(corrcoef(ac.T)[v_axis])[-2]  # last is autocorrelation

        accel = correct_accelerometer_orientation(accel, v_axis=v_axis, ap_axis=ap_axis)

    vert_accel = detrend(accel[:, v_axis])  # detrend data just in case

    # low-pass filter if we can
    if 0 < (2 * filter_cutoff / fs) < 1:
        sos = butter(filter_order, 2 * filter_cutoff / fs, btype="low", output="sos")
        # multiply by 1 to ensure a copy and not a view
        filt_vert_accel = sosfiltfilt(sos, vert_accel)
    else:
        filt_vert_accel = vert_accel * 1

    # first integrate the vertical accel to get velocity
    vert_velocity = cumtrapz(filt_vert_accel, x=ts - ts[0], initial=0)

    # get the CWT scales
    scale1, scale2 = get_cwt_scales(use_optimal_scale, vert_velocity, orig_scale, fs)

    coef1, _ = cwt(vert_velocity, [scale1, scale2], "gaus1")
    """
    Find the local minima in the signal. This should technically always require using
    the negative signal in "find_peaks", however the way PyWavelets computes the
    CWT results in the opposite signal that we want.
    Therefore, if the sign of the acceleration was negative, we need to use the
    positve coefficient signal, and opposite for positive acceleration reading.
    """
    init_contact, *_ = find_peaks(-va_sign * coef1[0], height=0.5 * std(coef1[0]))

    coef2, _ = cwt(coef1[1], scale2, "gaus1")
    """
    Peaks are the final contact points
    Same issue as above
    """
    final_contact, *_ = find_peaks(-va_sign * coef2[0], height=0.5 * std(coef2[0]))

    return init_contact, final_contact, filt_vert_accel, v_axis
