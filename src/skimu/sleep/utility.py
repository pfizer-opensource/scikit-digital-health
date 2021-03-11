"""
Utility functions required for sleep metric generation

Yiorgos Christakis
Pfizer DMTI 2021
"""
from numpy import any, asarray, append, where, cumsum, flip, arctan, pi, roll, abs, argmax, diff
from pandas import DataFrame

from skimu.utility import rolling_mean, rolling_sd

__all__ = [
    "detect_nonwear_mvmt", "detect_nonwear_temp", "rle", "rolling_median", "compute_z_angle",
    "compute_absolute_difference", "drop_min_blocks", "arg_longest_bout"
]


def detect_nonwear_mvmt(acc_rmed, fs, move_td=0.001):
    """
    Movement-based function for detecting non-wear.

    Parameters
    ----------
    acc_rmed : array
        5 second rolling mean of acceleration data
    fs : float
        Sampling frequency.
    move_td : float
        Movement threshold.

    Returns
    -------
    move_mask : array
        Epoch-level binary predictions of non-wear. 1 corresponds to a non-wear bout, 0 to a wear
        bout.

    """
    # rolling 5s mean (non-overlapping windows)
    mn = rolling_mean(acc_rmed, int(fs * 5), int(fs * 5), axis=0)

    # rolling 30m STD  5s windows -> 12 windows per minute
    rstd_mn = rolling_sd(mn, 12 * 30, 1, axis=0, return_previous=False)

    # threshold
    move_mask = any(rstd_mn <= move_td, axis=1)
    return move_mask


def detect_nonwear_temp(t, fs, temp_td=25.0):
    """
    Temperature-based function for detecting non-wear periods during sleep.

    Parameters
    ----------
    t : array
        Near-body temperature data.
    fs : float
        Sampling frequency.
    temp_td :
        Temperature threshold.

    Returns
    -------
    temp_mask : array
        Epoch-level binary predictions of non-wear. 1 corresponds to a non-wear bout, 0 to a wear bout.

    """
    # rolling 5s median
    rmd = rolling_median(t, int(fs * 5), 1).ravel()

    # rolling 5s mean (non-overlapping windows)
    mn = rolling_mean(rmd, int(fs * 5), int(fs * 5))

    # rolling 5m median.
    rmdn_mn = rolling_median(mn, 12 * 5, 1)

    # threshold
    temp_mask = rmdn_mn < temp_td
    return temp_mask


def rle(to_encode):
    """
    Run length encoding.

    Parameters
    ----------
    to_encode : array-like

    Returns
    -------
    lengths : array
        Lengths of each block.
    block_start_indices : array
        Indices of the start of each block.
    block_values : array
        The value repeated for the duration of each block.

    """
    array = asarray(to_encode)
    n = array.size

    is_diff = asarray(array[1:] != array[:-1])
    block_end_indices = append(where(is_diff), n - 1)
    lengths = diff(append(-1, block_end_indices))
    block_start_indices = cumsum(append(0, lengths))[:-1]
    block_values = array[block_start_indices]

    return lengths, block_start_indices, block_values


def _rolling_mean(arr, w_size, step=1):
    """
    Computes the rolling mean of an array along the first axis.

    Parameters
    ----------
    arr : array
    w_size : integer
        Window size in samples.
    step : integer
        Step size in samples.

    Returns
    -------
    rmn : array

    """
    df = DataFrame(flip(arr, axis=0))
    rmn = flip(df.rolling(w_size).mean().values, axis=0)[0: -(w_size - 1)][::step]
    return rmn


def rolling_median(arr, w_size, step=1):
    """
    Computes the rolling median of an array along the first axis.

    Parameters
    ----------
    arr : array
    w_size : integer
        Window size in samples.
    step : integer
        Step size in samples.

    Returns
    -------
    rmd : array

    """
    df = DataFrame(flip(arr, axis=0))
    rmd = flip(df.rolling(w_size).median().values, axis=0)[0: -(w_size - 1)][::step]
    return rmd


def _rolling_std(arr, w_size, step=1):
    """
    Computes the rolling standard deviation of an array along the first axis.

    Parameters
    ----------
    arr : array
    w_size : integer
        Window size in samples.
    step : integer
        Step size in samples.

    Returns
    -------
    rstd : array

    """
    df = DataFrame(flip(arr, axis=0))
    rstd = flip(df.rolling(w_size).std().values, axis=0)[0: -(w_size - 1)][::step]
    return rstd


def compute_z_angle(acc):
    """
    Computes the z-angle of a tri-axial accelerometer signal with columns X, Y, Z per sample.

    Parameters
    ----------
    acc : array

    Returns
    -------
    z : array

    """
    z = arctan(acc[:, 2] / ((acc[:, 0] ** 2 + acc[:, 1] ** 2) ** 0.5)) * (180.0 / pi)
    return z


def compute_absolute_difference(arr):
    """
    Computes the absolute difference between an array and itself shifted by 1 sample along the first axis.

    Parameters
    ----------
    arr : array

    Returns
    -------
    absd: array

    """
    shifted = roll(arr, 1)
    shifted[0] = shifted[1]
    absd = abs(arr - shifted)
    return absd


def drop_min_blocks(arr, min_block_size, drop_value, replace_value, skip_bounds=True):
    """
    Drops (rescores) blocks of a desired value with length less than some minimum length.
    (Ex. drop all blocks of value 1 with length < 5 and replace with new value 0).

    Parameters
    ----------
    arr : array
    min_block_size : integer
        Minimum acceptable block length in samples.
    drop_value : integer
        Value of blocks to examine.
    replace_value : integer
        Value to replace dropped blocks to.
    skip_bounds : boolean
        If True, ignores the first and last blocks.

    Returns
    -------
    arr : array

    """
    lengths, starts, vals = rle(arr)
    ctr = 0
    n = len(lengths)
    for length, start, val in zip(lengths, starts, vals):
        ctr += 1
        if skip_bounds and (ctr == 1 or ctr == n):
            continue
        if val == drop_value and length < min_block_size:
            arr[start: start + length] = replace_value
    return arr


def arg_longest_bout(arr, block_val):
    """
    Finds the first and last indices of the longest block of a given value present in a 1D array.

    Parameters
    ----------
    arr : array
        One-dimensional array.
    block_val : integer
        Value of the desired blocks.

    Returns
    -------
    longest_bout : tuple
        First, last indices of the longest block.

    """
    lengths, starts, vals = rle(arr)
    vals = vals.flatten()
    val_mask = vals == block_val
    if len(lengths[val_mask]):
        max_index = argmax(lengths[val_mask])
        max_start = starts[val_mask][max_index]
        longest_bout = max_start, max_start + lengths[val_mask][max_index]
    else:
        longest_bout = None, None
    return longest_bout
