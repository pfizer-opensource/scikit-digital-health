"""
Function for getting the bouts of continuous gait

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import where, diff, append, insert


def get_gait_bouts(bool_gait, dt, max_bout_separation, min_bout_time):
    """
    Get the gait bouts from an array of per sample predictions of gait

    Parameters
    ----------
    bool_gait : numpy.ndarray
        Array of boolean predictions of gait
    dt : float
        Sampling period
    max_bout_separation : float
        Maximum time (s) between bouts in order to consider them 1 longer bout
    min_bout_time : float
        Minimum time for a bout to be considered a bout of continuous gait

    Returns
    -------
    bouts : numpy.ndarray
        (M, 2) array of starts and stops of gait bouts
    """
    starts = where(diff(bool_gait.astype(int)) == 1)[0]
    stops = where(diff(bool_gait.astype(int)) == -1)[0]

    if bool_gait[0]:
        starts = insert(starts, 0, 0)
    if bool_gait[-1]:
        stops = append(stops, bool_gait.size - 1)

    if starts.size != stops.size:
        raise ValueError('Starts and stops of bouts do not match')

    bouts = []
    nb = 0
    while nb < starts.size:
        ncb = 0
        while ((starts[nb + ncb + 1] - stops[nb + ncb]) * dt) < max_bout_separation:
            ncb += 1

        if ((stops[nb + ncb] - starts[nb]) * dt) > min_bout_time:
            bouts.append((starts[nb], stops[nb + ncb]))

        nb += ncb + 1

    return bouts
