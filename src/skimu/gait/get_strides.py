"""
Function for getting strides from detected gait events

Lukas Adamowicz
Pfizer DMTI 2020
"""


def get_strides(gait, gait_index, ic, fc, dt, max_stride_time, loading_factor):
    """
    Get the strides from detected gait initial and final contacts, with optimizations

    Parameters
    ----------
    gait : dictionary
        Dictionary of gait values needed for computation or the results
    gait_index : int
        Where in the lists in `gait` the last bout left off
    ic : numpy.ndarray
        Indices of initial contact events
    fc : numpy.ndarray
        Indices of final contact events
    dt : float
        Sampling period
    max_stride_time : float
        Maximum time alloted for a stride
    loading_factor : float
        Factor to compute the maximum loading time for a stride

    Returns
    -------
    bout_n_steps : int
        Count of the number of steps/strides in the gait bout
    """
    loading_forward_time = loading_factor * max_stride_time
    stance_forward_time = (max_stride_time / 2) + loading_forward_time

    # create sample times for events
    ic_times = ic * dt
    fc_times = fc * dt

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
        bout_n_steps += 1

    if bout_n_steps > 2:
        gait['b valid cycle'].extend(
            [
                (
                    (gait['IC'][gait_index + i + 2] - gait['IC'][gait_index + i]) * dt
                ) < max_stride_time for i in range(bout_n_steps - 2)
            ]
        )
        gait['b valid cycle'].extend([False] * 2)
    elif bout_n_steps > 0:
        gait['b valid cycle'].extend([False] * bout_n_steps)

    return bout_n_steps
