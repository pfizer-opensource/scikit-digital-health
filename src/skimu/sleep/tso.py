"""
Function for the detection of sleep boundaries, here defined as the total sleep opportunity window.

Yiorgos Christakis
Pfizer DMTI 2021
"""
from numpy import pad, min, max, percentile, zeros, bool_

from skimu.utility import rolling_mean, rolling_median
from skimu.sleep.utility import *


def get_total_sleep_opportunity(
        fs, time, accel, wear_starts, wear_stops, min_rest_block, max_act_break,
        min_angle_thresh, max_angle_thresh, plot_fn, idx_start=0
):
    """
    Compute the period of time in which sleep can occur for a given days worth of data. For this
    algorithm, it is the longest period of wear-time that has low activity.

    Parameters
    ----------
    fs : float
        Sampling frequency of the time and acceleration data, in Hz.
    time : numpy.ndarray
        Timestamps for the acceleration.
    accel : numpy.ndarray
        (N, 3) array of acceleration values in g.
    wear_starts : numpy.ndarray
        Indices for the starts of wear-time. Note that while `time` and `accel` should be the values
        for one day, `wear_starts` is likely indexed to the whole data series. This offset can
        be adjusted by `idx_start`. If indexing only into the one day, `idx_start` should be 0.
    wear_stops : numpy.ndarray
        Indices for the stops of wear-time. Note that while `time` and `accel` should be the values
        for one day, `wear_stops` is likely indexed to the whole data series. This offset can
        be adjusted by `idx_start`. If indexing only into the one day, `idx_start` should be 0.
    min_rest_block : int
        Minimum number of minutes that a rest period can be
    max_act_break : int
        Maximum number of minutes an active block can be so that it doesn't interrupt a longer
        rest period.
    min_angle_thresh : float
        Minimum angle threshold used to compute the TSO threshold.
    max_angle_thresh : float
        Maximum angle threshold used to compute the TSO threshold.
    plot_fn : function
        Plotting function for the arm angle.
    idx_start : int, optional
        Offset index for wear-time indices. If `wear_starts` and `wear_stops` are relative to the
        day of interest, then `idx_start` should equal 0.

    Returns
    -------
    start : float
        Total sleep opportunity start timestamp.
    stop : float
        Total sleep opportunity stop timestamp.
    arg_start : int
        Total sleep opportunity start index, into the specific period of time.
    arg_stop : int
        Total sleep opportunity stop index, into the specific period of time.
    """
    # samples in 5 seconds
    n5 = int(5 * fs)
    # compute the rolling median for 5s windows
    acc_rmd = rolling_median(accel, n5, skip=1, pad=False, axis=0)

    # compute the z-angle
    z = compute_z_angle(acc_rmd)

    # rolling 5s mean with non-overlapping windows for the z-angle
    z_rm = rolling_mean(z, n5, n5)
    # plot arm angle
    plot_fn(z_rm)

    # the angle differences
    dz_rm = compute_absolute_difference(z_rm)

    # rolling 5 minute median. 12 windows per minute * 5 minutes
    dz_rm_rmd = rolling_median(dz_rm, 12 * 5, skip=1, pad=False)

    # compute the TSO threshold
    tso_thresh = compute_tso_threshold(dz_rm_rmd, min_td=min_angle_thresh, max_td=max_angle_thresh)

    # create the TSO mask (1 -> sleep opportunity, only happends during wear)
    tso = zeros(dz_rm_rmd.size, dtype=bool_)
    # block off nonwear times, scale by 5s blocks
    for strt, stp in zip((wear_starts - idx_start) / n5, (wear_stops - idx_start) / n5):
        tso[int(strt):int(stp)] = True

    # apply the threshold
    tso &= dz_rm_rmd < tso_thresh  # now only blocks where there is no movement, and wear are left

    # drop rest blocks less than minimum allowed rest length
    tso = drop_min_blocks(tso, 12 * min_rest_block, drop_value=1, replace_value=0)
    # drop active blocks less than maximum allowed active length
    tso = drop_min_blocks(tso, 12 * max_act_break, drop_value=0, replace_value=1)

    # get the indices of the longest bout
    arg_start, arg_end = arg_longest_bout(tso, 1)

    # get the timestamps of the longest bout
    if arg_start is not None:
        # account for left justified windows - times need to be bumped up by half a window
        # account for 5s windows in indexing
        arg_start = (arg_start + 30) * n5  # 12 * 5 / 2 = 30
        arg_end = (arg_end + 30) * n5

        start, end = time[arg_start], time[arg_end]
    else:
        start = end = None

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
