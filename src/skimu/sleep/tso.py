"""
Function for the detection of sleep boundaries, here defined as the total sleep opportunity window.

Yiorgos Christakis
Pfizer DMTI 2021
"""
from numpy import pad, min, max, percentile

from skimu.utility import rolling_mean
from skimu.sleep.utility import *


def detect_tso(
    acc,
    t,
    fs,
    temp=None,
    min_rest_block=30,
    allowed_rest_break=60,
    min_angle_threshold=0.1,
    max_angle_threshold=0.5,
    move_td=0.001,
    temp_td=25.0,
):
    """
    Computes the total sleep opportunity window bounds.

    Parameters
    ----------
    acc : array
        Tri-axial accelerometer data.
    t : array
        Timestamp array.
    fs : float
        Sampling frequency.
    temp : array
        Temperature data.
    min_rest_block : int
        Minimum number of minutes required to consider a rest period valid.
    allowed_rest_break : int
        Minimum number of minutes required to consider an active period valid.
    min_angle_threshold : float
        Minimum dz-angle threshold.
    max_angle_threshold : float
        Maximum dz-angle threshold.
    move_td : float
        Movement-based non-wear threshold value. Boolean False negates use.
    temp_td : float
        Temperature-based non-wear threshold value.

    Returns
    -------
    tso : tuple
        First and last timestamps of the tso window. First and last indices of tso window.

    """
    # compute rolling 5s median only 1 time
    rmd = rolling_median(acc, int(fs * 5), 1)

    # compute non-wear
    move_mask = detect_nonwear_mvmt(rmd, fs, move_td) if move_td else None
    temp_mask = detect_nonwear_temp(temp, fs, temp_td) if temp is not None else None

    # compute z-angle
    z = compute_z_angle(rmd)

    # rolling 5s mean (non-overlapping windows)
    mnz = rolling_mean(z, int(fs * 5), int(fs * 5))

    # compute dz-angle
    dmnz = compute_absolute_difference(mnz)

    # rolling 5m median
    rmd_dmnz = rolling_median(dmnz, 12 * 5, 1)  # 12 windows per minute (5s windows) * 5 minutes

    # compute threshold
    td = compute_tso_threshold(rmd_dmnz, min_td=min_angle_threshold, max_td=max_angle_threshold)

    # apply threshold
    rmd_dmnz[rmd_dmnz < td] = 0
    rmd_dmnz[rmd_dmnz >= td] = 1

    # apply movement-based non-wear mask
    if move_mask is not None:
        move_mask = pad(
            move_mask,
            (0, len(rmd_dmnz) - len(move_mask)),
            mode="constant",
            constant_values=1,
        )
        rmd_dmnz[move_mask] = 1

    # apply temperature-based non-wear mask
    if temp_mask is not None:
        temp_mask = pad(
            temp_mask,
            (0, len(rmd_dmnz) - len(temp_mask)),
            mode="constant",
            constant_values=1,
        )
        rmd_dmnz[temp_mask] = 1

    # drop rest blocks less than minimum allowed rest length
    rmd_dmnz = drop_min_blocks(rmd_dmnz, 12 * min_rest_block, drop_value=0, replace_value=1)

    # drop active blocks less than minimum allowed active length
    rmd_dmnz = drop_min_blocks(rmd_dmnz, 12 * allowed_rest_break, drop_value=1, replace_value=0)

    # get indices of longest bout
    arg_start, arg_end = arg_longest_bout(rmd_dmnz, 0)

    # get timestamps of longest bout
    t_5s = t[::int(5 * fs)]
    if arg_start is not None:
        start, end = t_5s[arg_start], t_5s[arg_end]
    else:
        start, end = None, None

    return start, end, arg_start, arg_end


def compute_tso_threshold(arr, min_td=0.1, max_td=0.5):
    """
    Computes the daily threshold value separating rest periods from active periods for the TSO detection algorithm.

    Parameters
    ----------
    arr : array
        Array of the absolute difference of the z-angle.
    min_td : float
        Minimum acceptable threshold value.
    max_td : float
        Maximum acceptable threshold value.

    Returns
    -------
    td : float

    """
    td = min((max((percentile(arr, 10) * 15.0, min_td)), max_td))
    return td
