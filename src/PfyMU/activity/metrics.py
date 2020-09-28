"""
Metric definitions

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import atan, sqrt, diff, cumsum, zeros, nanmedian, full, nan, where, isnan
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt

from PfyMU.features import get_windowed_view


__all__ = ['roll_mean', 'roll_median_1', 'angle', 'hfen_plus', 'enmo', 'enmoa']


def roll_mean(x, fs, win_s):
    """
    Compute the mean across non-overlapping windows

    Parameters
    ----------
    x : numpy.ndarray
        signal
    fs : float
        sampling frequency
    win_s : float
        window size in seconds
    """
    n = int(round(fs * win_s))
    x2 = zeros(x.size + 1)
    x2[1:] = cumsum(x)
    return diff(x2[::n]) / n


def roll_median_1(x, fs, win_s=5.0):
    """
    Compute the rolling median across maximally overlapping windows

    Parameters
    ----------
    x : numpy.ndarray
        signal
    fs : float,
        sampling frequency
    win_s : float
        window size in seconds
    """
    n = int(round(fs * win_s))
    n2 = int(n // 2)
    xw = get_windowed_view(x, n, 1)

    xm = full(x.shape, nan)
    xm[n2:-n2] = nanmedian(xw, axis=-1)
    # GGIR uses median, replaces NaN values with the first non nan value in the whole array

    # replace NaN values with closest value on the beginning and end (to make returned array
    # same shape as input)
    ind = where(~isnan(xm))[0]
    first, last = ind[0], ind[-1]
    xm[:first] = xm[first]
    xm[last + 1:] = xm[last]

    return xm


def angle(acc, axis):
    """
    Compute the angle of the sensor based on the acceleration data along the specified axis.

    Parameters
    ----------
    acc : numpy.ndarray
        (N, 3) array of acceleration data
    axis : {'x', 'y', 'z'}
        Axis to compute the angle for

    Returns
    -------
    angle : numpy.ndarray
        (N, ) array of angles
    """
    if axis == 'x':
        return atan(acc[:, 0] / sqrt(acc[:, 1]**2 + acc[:, 2]**2))
    elif axis == 'y':
        return atan(acc[:, 1] / sqrt(acc[:, 0]**2 + acc[:, 2]**2))
    elif axis == 'z':
        return atan(acc[:, 2] / sqrt(acc[:, 1]**2 + acc[:, 0]**2))
    else:
        raise ValueError("'axis' must be in {'x', 'y', 'z'}")


def hfen_plus(acc, fs, cut=0.2, N=4):
    """
    Compute the High-pass Filter with Euclidean norm, plus the low-pass euclidean norm less
    gravitiy

    Parameters
    ----------
    acc : numpy.ndarray
        (N, 3) ndarray of accelerations in units of g
    fs : float
        Sampling frequency
    cut : float, optional
        Cutoff for both the high- and low-pass filters. Default is 0.2Hz
    N : int, optional
        Filter order. Default is 4

    Returns
    -------
    hfenp : numpy.ndarray
        (N, ) array of the HFEN+ metric
    """
    sos_hip = butter(N, 2 * cut / fs, btype='high')
    sos_lop = butter(N, 2 * cut / fs, btype='low')

    tmp_hi = sosfiltfilt(sos_hip, acc, axis=0)
    tmp_lo = sosfiltfilt(sos_lop, acc, axis=0)

    return norm(tmp_hi, axis=1) + norm(tmp_lo, axis=1) - 1  # minus gravity


def enmo(acc):
    """
    Compute the Euclidean Norm minus gravity

    Parameters
    ----------
    acc : numpy.ndarray
        (N, 3) array of accelerations in units of g

    Returns
    -------
    enmo : numpy.ndarray
        (N, ) array of the ENMO metric
    """
    return norm(acc, axis=1) - 1


def enmoa(acc):
    """
    Compute the Euclidean Norm minus gravity, with negative values clipped to 0

    Parameters
    ----------
    acc : numpy.ndarray
        (N, 3) array of accelerations in units of g

    Returns
    -------
    enmoa : numpy.ndarray
        (N, ) array of the ENMOa metric
    """
    tmp = norm(acc, axis=1) - 1
    tmp[tmp < 0] = 0.0
    return tmp
