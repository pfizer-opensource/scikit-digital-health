"""
Utility functions required for sleep metric generation

Yiorgos Christakis
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from numpy import (
    any,
    arctan,
    pi,
    roll,
    abs,
    argmax,
    diff,
    nonzero,
    insert,
    sqrt,
    pad,
    int_,
    append,
    mean,
    var,
    ascontiguousarray,
)
from scipy.signal import butter, sosfiltfilt

from skdh.utility import get_windowed_view
from skdh.utility import moving_mean, moving_sd, moving_median
from skdh.utility.internal import rle

__all__ = [
    "compute_z_angle",
    "compute_absolute_difference",
    "drop_min_blocks",
    "arg_longest_bout",
    "compute_activity_index",
]


def get_weartime(acc_rmed, temp, fs, move_thresh, temp_thresh):
    """
    Compute the wear time using acceleration and temperature data.

    Parameters
    ----------
    acc_rmed : numpy.ndarray
        Rolling median acceleration with 5s windows and 1 sample skips.
    temp : numpy.ndarray
        Raw temperature data.
    fs : float
        Sampling frequency.
    move_thresh : float
        Threshold to classify acceleration as wear/nonwear
    temp_thresh : float
        Temperature threshold to classify as wear/nonwear

    Returns
    -------
    wear : numpy.ndarray
        (N, 2) array of [start, stop] indices of blocks of wear time.
    """
    n5 = int(5 * fs)
    # rolling 5s mean (non-overlapping windows)
    mn = moving_mean(acc_rmed, n5, n5, axis=0)
    # rolling 30min StDev.  5s windows -> 12 windows per minute
    acc_rsd = moving_sd(mn, 12 * 30, 1, axis=0, return_previous=False)
    # TODO note that this 30 min rolling standard deviation likely means that our wear/nonwear
    # timest could be off by as much as 30 mins, due to windows extending into the wear time.
    # this is likely going to be an issue for all wear time algorithms due to long
    # windows, however.

    # rolling 5s median of temperature
    rmd = moving_median(temp, n5, skip=1)
    # rolling 5s mean (non-overlapping)
    mn = moving_mean(rmd, n5, n5)
    # rolling 5m median
    temp_rmd = moving_median(mn, 12 * 5, skip=1)

    move_mask = any(acc_rsd > move_thresh, axis=1)
    temp_mask = temp_rmd >= temp_thresh

    # pad the movement mask, temperature mask is the correct size
    npad = temp_mask.size - move_mask.size
    move_mask = pad(move_mask, (0, npad), mode="constant", constant_values=0)

    dwear = diff((move_mask | temp_mask).astype(int_))

    starts = nonzero(dwear == 1)[0] + 1
    stops = nonzero(dwear == -1)[0] + 1

    if move_mask[0] or temp_mask[0]:
        starts = insert(starts, 0, 0)
    if move_mask[-1] or temp_mask[-1]:
        stops = append(stops, move_mask.size)

    return starts * n5, stops * n5


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
    z = arctan(acc[:, 2] / sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2)) * (180.0 / pi)
    return z


def compute_absolute_difference(arr):
    """
    Computes the absolute difference between an array and itself shifted by 1 sample along the
    first axis.

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
            arr[start : start + length] = replace_value
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


def compute_activity_index(fs, accel, hp_cut=0.25):
    """
    Calculate the activity index

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    accel : numpy.ndarray
        Acceleration
    hp_cut : float
        High-pass filter cutoff

    Returns
    -------
    ai : numpy.ndarray
        The activity index of non-overlapping 60s windows
    """
    # high pass filter
    sos = butter(3, hp_cut * 2 / fs, btype="high", output="sos")
    accel_hf = ascontiguousarray(sosfiltfilt(sos, accel, axis=0))

    # non-overlapping 60s windows
    acc_w = get_windowed_view(accel_hf, int(60 * fs), int(60 * fs))

    # compute activity index
    act_ind = sqrt(mean(var(acc_w, axis=2), axis=1))

    return act_ind
