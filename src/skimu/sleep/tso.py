"""
Function for the detection of sleep boundaries, here defined as the total sleep opportunity window.

Yiorgos Christakis
Pfizer DMTI 2021
"""
from src.skimu.sleep.utility import *


def detect_tso(
    acc,
    t,
    fs,
    temp=None,
    min_rest=30,
    min_act=60,
    min_td=0.1,
    max_td=0.5,
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
    min_rest : int
        Minimum number of minutes required to consider a rest period valid.
    min_act : int
        Minimum number of minutes required to consider an active period valid.
    min_td : float
        Minimum dz-angle threshold.
    max_td : float
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

    # upsample/downsample to 20hz if necessary
    if fs != 20.0:
        # get timestamps
        t_ds = np.arange(t[0], t[-1], 1 / 20.0)

        # get acceleration
        acc_ds = np.zeros((t_ds.size, 3), dtype=np.float_)
        for i in range(3):
            acc_ds[:, i] = np.interp(t_ds, t, acc[:, i])
        acc = acc_ds

        # get temp
        if temp is not None:
            temp_ds = np.zeros((t_ds.size, 1), dtype=np.float_)
            temp_ds[:, 0] = np.interp(t_ds, t, temp)
            temp = temp_ds

        # reset time
        t = t_ds
    fs = 20

    # compute non-wear
    move_mask = detect_nonwear_mvmt(acc, fs, move_td) if move_td else None
    temp_mask = detect_nonwear_temp(temp, fs, temp_td) if temp is not None else None

    # rolling 5s median
    rmd = rolling_median(acc, fs * 5, 1)

    # compute z-angle
    z = compute_z_angle(rmd)

    # rolling 5s mean (non-overlapping windows)
    mnz = rolling_mean(z, fs * 5, fs * 5)

    # compute dz-angle
    dmnz = compute_absolute_difference(mnz)

    # rolling 5m median
    rmd_dmnz = rolling_median(dmnz, 5 * 12 * 5, 1)

    # compute threshold
    td = compute_tso_threshold(rmd_dmnz, min_td=min_td, max_td=max_td)

    # apply threshold
    rmd_dmnz[rmd_dmnz < td] = 0
    rmd_dmnz[rmd_dmnz >= td] = 1

    # apply movement-based non-wear mask
    if move_mask is not None:
        move_mask = np.pad(
            move_mask,
            (0, len(rmd_dmnz) - len(move_mask)),
            mode="constant",
            constant_values=1,
        )
        rmd_dmnz[move_mask] = 1

    # apply temperature-based non-wear mask
    if temp_mask is not None:
        temp_mask = np.pad(
            temp_mask,
            (0, len(rmd_dmnz) - len(temp_mask)),
            mode="constant",
            constant_values=1,
        )
        rmd_dmnz[temp_mask] = 1

    # drop rest blocks less than minimum allowed rest length
    rmd_dmnz = drop_min_blocks(rmd_dmnz, 12 * min_rest, drop_value=0, replace_value=1)

    # drop active blocks less than minimum allowed active length
    rmd_dmnz = drop_min_blocks(rmd_dmnz, 12 * min_act, drop_value=1, replace_value=0)

    # get indices of longest bout
    arg_start, arg_end = arg_longest_bout(rmd_dmnz, 0)

    # get timestamps of longest bout
    t_5s = t[:: 5 * fs]
    if arg_start is not None:
        start, end = t_5s[arg_start], t_5s[arg_end]
    else:
        start, end = None, None

    tso = (start, end, arg_start, arg_end)
    return tso


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
    td = np.min((np.max((np.percentile(arr, 10) * 15.0, min_td)), max_td))
    return td


def check_tso_detection():
    from skimu.read import ReadBin

    src = "/Users/ladmin/Desktop/PfyMU_development/sleeppy_pfymu/test_data/demo.bin"
    reader = ReadBin(base=12, period=24)
    res = reader.predict(src)
    for day in res["day_ends"]:
        acc = res["accel"][day[0] : day[1]]
        t = res["time"][day[0] : day[1]]
        fs = 100
        temp = res["temperature"][day[0] : day[1]]
        temp = np.repeat(temp, 300)
        out = detect_tso(acc=acc, t=t, fs=fs, temp=temp)
    print(out)
    print(pd.to_datetime(out[0], unit="s"), pd.to_datetime(out[1], unit="s"))

