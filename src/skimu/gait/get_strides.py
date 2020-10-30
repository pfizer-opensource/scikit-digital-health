"""
Function for getting strides from detected gait events

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import nan, array
from scipy.signal import detrend
from scipy.integrate import cumtrapz


def get_strides(gait, vert_accel, gait_index, ic, fc, timestamps, max_stride_time, loading_factor):
    """
    Get the strides from detected gait initial and final contacts, with optimizations

    Parameters
    ----------
    gait : dictionary
        Dictionary of gait values needed for computation or the results
    vert_accel numpy.ndarray
        (N, ) array of vertial acceleration
    gait_index : int
        Where in the lists in `gait` the last bout left off
    ic : numpy.ndarray
        Indices of initial contact events
    fc : numpy.ndarray
        Indices of final contact events
    timestamps : numpy.ndarray
        Timestamps (in seconds) corresponding to the acceleration
    max_stride_time : float
        Maximum time alloted for a stride
    loading_factor : float
        Factor to compute the maximum loading time for a stride

    Returns
    -------
    bout_n_steps : int
        Count of the number of steps/strides in the gait bout
    """
    assert vert_accel.size == timestamps.size, "`vert_accel` and `timestamps` size must match"

    loading_forward_time = loading_factor * max_stride_time
    stance_forward_time = (max_stride_time / 2) + loading_forward_time

    # create sample times for events
    ic_times = timestamps[ic]
    fc_times = timestamps[fc]

    # for easier use later
    gait_ic_times = []

    bout_n_steps = 0  # steps in bout
    for i, curr_ic in enumerate(ic_times):
        fc_forward = fc[fc_times > curr_ic]
        fc_forward_times = fc_times[fc_times > curr_ic]

        # OPTIMIZATION 1: initial double support (loading) time should be less than
        # max_stride_time * loading_factor
        if (fc_forward_times < (curr_ic + loading_forward_time)).sum() != 1:
            continue  # skip this IC
        # OPTIMIZATION 2: stance time should be less than half a gait cycle
        # + initial double support time
        if (fc_forward_times < (curr_ic + stance_forward_time)).sum() < 2:
            continue  # skip this ic

        # if this point is reached, both optimizations passed
        gait['IC'].append(ic[i])
        gait['FC'].append(fc_forward[1])
        gait['FC opp foot'].append(fc_forward[0])
        gait_ic_times.append(ic_times[i])
        bout_n_steps += 1

    # convert to array for vector subtraction
    gait_ic_times = array(gait_ic_times)

    if bout_n_steps > 2:
        gait['b valid cycle'].extend(gait_ic_times[2:] - gait_ic_times[:-2] < max_stride_time)
        gait['b valid cycle'].extend([False] * 2)
    elif bout_n_steps > 0:
        gait['b valid cycle'].extend([False] * bout_n_steps)

    for i in range(gait_index, gait_index + bout_n_steps - 1):
        i1 = gait['IC'][i]
        i2 = gait['IC'][i+1]

        if gait['b valid cycle'][i]:
            vacc = detrend(vert_accel[i1:i2])
            vvel = cumtrapz(vacc, x=timestamps[i1:i2], initial=0)
            vpos = cumtrapz(vvel, x=timestamps[i1:i2], initial=0)

            gait['delta h'].append((vpos.max() - vpos.min()) * 9.81)  # convert to meters
        else:
            gait['delta h'].append(nan)

    # make sure parameters here math the number of steps in gait
    gait['delta h'].append(nan)

    return bout_n_steps
