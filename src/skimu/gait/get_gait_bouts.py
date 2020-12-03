"""
Function for getting the bouts of continuous gait

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import insert, append


def get_gait_bouts(starts, stops, day_start, day_stop, timestamps, max_bout_separation,
                   min_bout_time):
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
    timestamps : numpy.ndarray
        Array of timestmaps (in seconds) corresponding to acceleration sampling times.
    max_bout_separation : float
        Maximum time (s) between bouts in order to consider them 1 longer bout
    min_bout_time : float
        Minimum time for a bout to be considered a bout of continuous gait

    Returns
    -------
    bouts : numpy.ndarray
        (M, 2) array of starts and stops of gait bouts
    """
    # get the portion of bouts for the specified day
    starts_subset = starts[(starts > day_start) & (starts < day_stop)]
    stops_subset = stops[(stops > day_start) & (stops < day_stop)]

    if starts_subset[0] > stops_subset[0]:
        starts_subset = insert(starts_subset, 0, day_start)
    if starts_subset[-1] > stops_subset[-1]:
        stops_subset = append(stops_subset, day_stop)

    assert starts_subset.size == stops_subset.size, 'Starts and stops of bouts do not match'

    bouts = []
    nb = 0
    while nb < starts_subset.size:
        ncb = 0

        if (nb + ncb + 1) < starts_subset.size:
            # should be start - stop because it is start_(i+1)
            tdiff = timestamps[starts_subset[nb + ncb + 1]] - timestamps[stops_subset[nb + ncb]]
            while (tdiff < max_bout_separation) and ((nb + ncb + 2) < starts_subset.size):
                ncb += 1
                # should be start - stop because it is start_(i+1)
                tdiff = timestamps[starts_subset[nb + ncb + 1]] - timestamps[stops_subset[nb + ncb]]

        if (timestamps[stops_subset[nb + ncb]-1] - timestamps[starts_subset[nb]]) > min_bout_time:
            bouts.append(slice(starts_subset[nb], stops_subset[nb + ncb]))

        nb += ncb + 1

    return bouts
