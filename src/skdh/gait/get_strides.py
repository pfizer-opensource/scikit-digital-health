"""
Function for getting strides from detected gait events

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import nan, array, ones, nonzero, zeros
from scipy.integrate import cumtrapz


def get_strides(
    gait, vert_accel, gait_index, ic, fc, ts, fs, max_stride_time, loading_factor
):
    """
    Get the strides from detected gait initial and final contacts, with optimizations

    Parameters
    ----------
    gait : dictionary
        Dictionary of gait values needed for computation or the results
    vert_accel : numpy.ndarray
        (N, ) array of vertial acceleration
    gait_index : int
        Where in the lists in `gait` the last bout left off
    ic : numpy.ndarray
        Indices of initial contact events
    fc : numpy.ndarray
        Indices of final contact events
    ts : numpy.ndarray
        Timestamps (in seconds) corresponding to the acceleration
    fs : float
        Sampling frequency for the acceleration/timestamps.
    max_stride_time : float
        Maximum time alloted for a stride
    loading_factor : float
        Factor to compute the maximum loading time for a stride

    Returns
    -------
    bout_n_steps : int
        Count of the number of steps/strides in the gait bout
    """
    assert vert_accel.size == ts.size, "`vert_accel` and `ts` size must match"

    loading_forward_time = loading_factor * max_stride_time
    stance_forward_time = (max_stride_time / 2) + loading_forward_time

    # create sample times for events
    ic_times = ts[ic]
    fc_times = ts[fc]

    # for easier use later
    gait_ic_times = []
    gait_fc_times = []
    gait_fc_opp_times = []

    # mask the steps that have already been used, to avoid using duplicate FC events
    fc_unused = ones(fc_times.size, dtype="bool")

    bout_n_steps = 0
    for i, curr_ic in enumerate(ic_times):
        forward_idx = nonzero((fc_times > curr_ic) & fc_unused)[0]
        fc_forward = fc[forward_idx]
        fc_forward_times = fc_times[forward_idx]

        # OPTIMIZATION 1: initial double support (loading) time should be less than
        # max_stride_time * loading_factor
        if (fc_forward_times < (curr_ic + loading_forward_time)).sum() != 1:
            continue  # skip this IC
        # OPTIMIZATION 2: stance time should be less than half gait cycle + initial double support
        if (fc_forward_times < (curr_ic + stance_forward_time)).sum() < 2:
            continue  # skip this IC

        # if this point is reached, both optimizations passed
        gait["IC"].append(ic[i])
        gait["FC"].append(fc_forward[1])
        gait["FC opp foot"].append(fc_forward[0])
        gait["IC Time"].append(curr_ic)
        gait_ic_times.append(ic_times[i])
        gait_fc_times.append(fc_forward_times[1])
        gait_fc_opp_times.append(fc_forward_times[0])

        # block off these FCs from being used for future steps
        # We only need to block off the opp foot FC because the FC
        # for this step will become FC opp foot for the next step
        fc_unused[forward_idx[0]] = False

        bout_n_steps += 1

    # convert to array for vector subtration
    gait_ic_times = array(gait_ic_times)
    gait_fc_times = array(gait_fc_times)
    gait_fc_opp_times = array(gait_fc_opp_times)

    forward_cycles = zeros(gait_ic_times.size, dtype="int")
    # are there 2 forward cycles within the maximum stride time
    forward_cycles[:-2] += (gait_ic_times[2:] - gait_ic_times[:-2]) < max_stride_time
    # is the next step continuous
    forward_cycles[:-1] += gait_fc_opp_times[1:] == gait_fc_times[:-1]

    gait["forward cycles"].extend(forward_cycles)

    for i in range(gait_index, gait_index + bout_n_steps - 1):
        i1 = gait["IC"][i]
        i2 = gait["IC"][i + 1]

        if gait["forward cycles"][i] > 0:
            vacc = vert_accel[i1:i2]
            vvel = cumtrapz(vacc, x=ts[i1:i2], initial=0)
            vpos = cumtrapz(vvel, x=ts[i1:i2], initial=0)

            gait["delta h"].append(
                (vpos.max() - vpos.min()) * 9.81
            )  # convert to meters
        else:
            gait["delta h"].append(nan)

    # make sure parameters here match the number of steps in gait
    gait["delta h"].extend([nan] * min(1, bout_n_steps))

    return bout_n_steps
