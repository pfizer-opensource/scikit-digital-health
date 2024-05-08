"""
Function for getting the bouts of continuous gait

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from skdh.utility.internal import get_day_index_intersection


def get_gait_bouts(starts, stops, day_start, day_stop, ts, max_bout_sep, min_bout_time):
    """
    Get the gait bouts from an array of per sample predictions of gait

    Parameters
    ----------
    starts : numpy.ndarray
        Array of integer indices where gait bouts start
    stops : numpy.ndarray
        Array of integer indices where gait bouts end
    day_start : integer
        Index of the day start
    day_stop : integer
        Index of the day stop
    ts : numpy.ndarray
        Array of timestmaps (in seconds) corresponding to acceleration sampling times.
    max_bout_sep : float
        Maximum time (s) between bouts in order to consider them 1 longer bout
    min_bout_time : float
        Minimum time for a bout to be considered a bout of continuous gait

    Returns
    -------
    bouts : list
     List slices with the starts and stops of gait bouts
    """
    starts_subset, stops_subset = get_day_index_intersection(
        (starts,), (stops,), (True,), day_start, day_stop
    )

    bouts = []
    nb = 0
    while nb < starts_subset.size:
        ncb = 0  # number continuous bouts

        if (nb + ncb + 1) < starts_subset.size:
            # should be start - stop because it is start_(i+1)
            tdiff = ts[starts_subset[nb + ncb + 1]] - ts[stops_subset[nb + ncb]]
            while (tdiff < max_bout_sep) and ((nb + ncb + 2) < starts_subset.size):
                ncb += 1
                tdiff = ts[starts_subset[nb + ncb + 1]] - ts[stops_subset[nb + ncb]]

        if (ts[stops_subset[nb + ncb] - 1] - ts[starts_subset[nb]]) > min_bout_time:
            bouts.append(slice(starts_subset[nb], stops_subset[nb + ncb]))

        nb += ncb + 1

    return bouts
