"""
Metrics for classifying activity

Lukas Adamowicz
Pfizer DMTI 2021
"""
from numpy import minimum, abs, repeat
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt

from skimu.utility import rolling_mean


__all__ = ["metric_en", "metric_enmo", "metric_bfen", "metric_hfen", "metric_hfenplus"]


def metric_en(accel, wlen, *args, **kwargs):
    """
    Euclidean norm

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    wlen : int
        Window length (in number of samples) for non-overlapping windows.

    Returns
    -------
    en : numpy.ndarray
        (N, ) array of euclidean norms.
    """
    return rolling_mean(norm(accel, axis=1), wlen, wlen)


def metric_enmo(accel, wlen, *args, take_abs=False, trim_zero=True, **kwargs):
    """
    Euclidean norm minus 1. Works best when the accelerometer data has been calibrated so that
    at rest the norm meaures 1g.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    wlen : int
        Window length (in number of samples) for non-overlapping windows.
    take_abs : bool, optional
        Use the absolute value of the difference between euclidean norm and 1g. Default is False.
    trim_zero : bool, optional
        Trim values to no less than 0. Default is True.

    Returns
    -------
    enmo : numpy.ndarray
        (N, ) array of euclidean norms minus 1.
    """
    enmo = norm(accel, axis=1) - 1
    if take_abs:
        enmo = abs(enmo)
    if trim_zero:
        return rolling_mean(minimum(enmo, 0), wlen, wlen)
    else:
        return rolling_mean(enmo, wlen, wlen)


def metric_bfen(accel, wlen, fs, low_cutoff=0.2, high_cutoff=15, trim_zero=True, **kwargs):
    """
    Band-pass filtered euclidean norm.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    wlen : int
        Window length (in number of samples) for non-overlapping windows.
    fs : float
        Sampling frequency of `accel` in Hz.
    low_cutoff : float, optional
        Band-pass low cutoff in Hz. Default is 0.2Hz.
    high_cutoff : float, optional
        Band-pass high cutoff in Hz. Default is 15Hz
    trim_zero : bool, optional
        Trim values to no less than 0. Default is True.

    Returns
    -------
    bfen : numpy.ndarray
        (N, ) array of band-pass filtered and euclidean normed accelerations.
    """
    sos = butter(4, [2 * low_cutoff / fs, 2 * high_cutoff / fs], btype='bandpass', output='sos')
    if trim_zero:
        return rolling_mean(minimum(norm(sosfiltfilt(sos, accel, axis=0), axis=1), 0), wlen, wlen)
    else:
        return rolling_mean(norm(sosfiltfilt(sos, accel, axis=0), axis=1), wlen, wlen)


def metric_hfen(accel, wlen, fs, low_cutoff=0.2, trim_zero=True, **kwargs):
    """
    High-pass filtered euclidean norm.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    wlen : int
        Window length (in number of samples) for non-overlapping windows.
    fs : float
        Sampling frequency of `accel` in Hz.
    low_cutoff : float, optional
        High-pass cutoff in Hz. Default is 0.2Hz.
    trim_zero : bool, optional
        Trim values to no less than 0. Default is True.

    Returns
    -------
    hfen : numpy.ndarray
        (N, ) array of high-pass filtered and euclidean normed accelerations.
    """
    sos = butter(4, 2 * low_cutoff / fs, btype='high', output='sos')
    
    if trim_zero:
        return rolling_mean(minimum(norm(sosfiltfilt(sos, accel, axis=0), axis=1), 0), wlen, wlen)
    else:
        return rolling_mean(norm(sosfiltfilt(sos, accel, axis=0), axis=1), wlen, wlen)


def metric_hfenplus(accel, wlen, fs, cutoff=0.2, trim_zero=True, **kwargs):
    """
    High-pass filtered euclidean norm plus the low-pass filtered euclidean norm minus 1g.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    wlen : int
        Window length (in number of samples) for non-overlapping windows.
    fs : float
        Sampling frequency of `accel` in Hz.
    cutoff : float, optional
        Cutoff in Hz for both high and low filters. Default is 0.2Hz.
    trim_zero : bool, optional
        Trim values to no less than 0. Default is True.

    Returns
    -------
    hfenp : numpy.ndarray
        (N, ) array of high-pass filtered acceleration norm added to the low-pass filtered
        norm minus 1g.
    """
    sos_low = butter(4, 2 * cutoff / fs, btype="low", output="sos")
    sos_high = butter(4, 2 * cutoff / fs, btype="high", output="sos")

    acc_high = norm(sosfiltfilt(sos_high, accel, axis=0), axis=1)
    acc_low = norm(sosfiltfilt(sos_low, accel, axis=0), axis=1)
    
    if trim_zero:
        return rolling_mean(minimum(acc_high + acc_low - 1, 0), wlen, wlen)
    else:
        return rolling_mean(acc_high + acc_low - 1, wlen, wlen)


def metric_mad(accel, wlen, *args, **kwargs):
    """
    Compute the Mean Amplitude Deviation metric for acceleration.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of accelerationes measured in g.
    wlen : int
        Window length (in number of samples) for non-overlapping windows.

    Returns
    -------
    mad : numpy.ndarray
        (N, ) array of computed MAD values.
    """
    acc_norm = norm(accel, axis=1)
    r_avg = repeat(rolling_mean(acc_norm, wlen, wlen), wlen)

    mad = rolling_mean(abs(acc_norm[:r_avg.size] - r_avg), wlen, wlen)
    return mad
