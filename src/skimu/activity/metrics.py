"""
Metrics for classifying activity

Lukas Adamowicz
Pfizer DMTI 2021
"""
from numpy import minimum
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt


def metric_en(accel):
    """
    Euclidean norm

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.

    Returns
    -------
    en : numpy.ndarray
        (N, ) array of euclidean norms.
    """
    return norm(accel, axis=1)


def metric_enmo(accel, round_zero=True):
    """
    Euclidean norm minus 1. Works best when the accelerometer data has been calibrated so that
    at rest the norm meaures 1g.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    round_zero : bool, optional
        Trim values to no less than 0. Default is True.

    Returns
    -------
    enmo : numpy.ndarray
        (N, ) array of euclidean norms minus 1.
    """
    return minimum(norm(accel, axis=1) - 1, 0)


def metric_bfen(accel, fs, low_cutoff=0.2, high_cutoff=15):
    """
    Band-pass filtered euclidean norm.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    fs : float
        Sampling frequency of `accel` in Hz.
    low_cutoff : float, optional
        Band-pass low cutoff in Hz. Default is 0.2Hz.
    high_cutoff : float, optional
        Band-pass high cutoff in Hz. Default is 15Hz

    Returns
    -------
    bfen : numpy.ndarray
        (N, ) array of band-pass filtered and euclidean normed accelerations.
    """
    sos = butter(4, [2 * low_cutoff / fs, 2 * high_cutoff / fs], btype='bandpass', output='sos')
    return norm(sosfiltfilt(sos, accel, axis=0), axis=1)


def metric_hfen(accel, fs, low_cutoff=0.2):
    """
    High-pass filtered euclidean norm.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    fs : float
        Sampling frequency of `accel` in Hz.
    low_cutoff : float, optional
        High-pass cutoff in Hz. Default is 0.2Hz.

    Returns
    -------
    hfen : numpy.ndarray
        (N, ) array of high-pass filtered and euclidean normed accelerations.
    """
    sos = butter(4, 2 * low_cutoff / fs, btype='high', output='sos')
    return norm(sosfiltfilt(sos, accel, axis=0), axis=1)


def metric_hfenplus(accel, fs, cutoff=0.2):
    """
    High-pass filtered euclidean norm plus the low-pass filtered euclidean norm minus 1g.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    fs : float
        Sampling frequency of `accel` in Hz.
    cutoff : float, optional
        Cutoff in Hz for both high and low filters. Default is 0.2Hz.

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
    return acc_high + acc_low - 1
