"""
Internal utility functions that don't necessarily need to be exposed in the public API

Lukas Adamowicz
Pfizer DMTI 2021
"""
from numpy import array, nonzero, insert, append, arange, interp, zeros, around, float_, int_


def get_day_wear_intersection(starts, stops, day_start, day_stop):
    """
    Get the intersection between wear times and day starts/stops.
    Parameters
    ----------
    starts : numpy.ndarray
        Array of integer indices where gait bouts start.
    stops : numpy.ndarray
        Array of integer indices where gait bouts stop.
    day_start : int
        Index of the day start.
    day_stop : int
        Index of the day stop.
    Returns
    -------
    day_wear_starts : numpy.ndarray
        Array of wear start indices for the day.
    day_wear_stops : numpy.ndarray
        Array of wear stop indices for the day
    """
    day_start, day_stop = int(day_start), int(day_stop)
    # get the portion of wear times for the day
    starts_subset = starts[(starts >= day_start) & (starts < day_stop)]
    stops_subset = stops[(stops > day_start) & (stops <= day_stop)]

    if starts_subset.size == 0 and stops_subset.size == 0:
        try:
            if stops[nonzero(starts <= day_start)[0][-1]] >= day_stop:
                return array([day_start]), array([day_stop])
            else:
                return array([]), array([])
        except IndexError:
            return array([]), array([])
    if starts_subset.size == 0 and stops_subset.size == 1:
        starts_subset = array([day_start])
    if starts_subset.size == 1 and stops_subset.size == 0:
        stops_subset = array([day_stop])

    if starts_subset[0] > stops_subset[0]:
        starts_subset = insert(starts_subset, 0, day_start)
    if starts_subset[-1] > stops_subset[-1]:
        stops_subset = append(stops_subset, day_stop)

    assert starts_subset.size == stops_subset.size, "bout starts and stops do not match"

    return starts_subset, stops_subset


def apply_downsample(goal_fs, time, data=(), indices=()):
    """
    Apply a downsample to a set of data.

    Parameters
    ----------
    goal_fs : float
        Desired sampling frequency in Hz.
    time : numpy.ndarray
        Array of original timestamps.
    data : tuple, optional
        Tuple of arrays to normally downsample using interpolation. Must match the size of `time`.
        Can handle `None` inputs, and will return an array of zeros matching the downsampled size.
    indices : tuple, optional
        Tuple of arrays of indices to downsample.

    Returns
    -------
    time_ds : numpy.ndarray
        Downsampled time.
    data_ds : tuple, optional
        Downsampled data, if provided.
    indices : tuple, optional
        Downsampled indices, if provided.
    """
    time_ds = arange(time[0], time[-1], 1 / goal_fs)

    data_ds = ()

    for dat in data:
        if dat is None:
            data_ds += (zeros(time_ds.size, dtype=float_))
        elif dat.ndim == 1:
            data_ds += (interp(time_ds, time, dat))
        elif dat.ndim == 2:
            data_ds += (zeros((time_ds.size, dat.shape[1]), dtype=float_),)
            for i in range(dat.shape[1]):
                data_ds[-1][:, i] = interp(time_ds, time, dat[:, i])
        else:
            raise ValueError("Data dimension exceeds 2, or data not understood.")

    # downsampling indices
    indices_ds = ()
    for idx in indices:
        if idx.ndim == 1:
            indices_ds += (
                around(
                    interp(time[idx], time_ds, arange(time_ds.size))
                ).astype(int_),
            )
        elif idx.ndim == 2:
            indices_ds += (zeros(idx.shape, dtype=int_),)
            for i in range(idx.shape[1]):
                indices_ds[-1][:, i] = around(
                    interp(time[idx[:, i]], time_ds, arange(time_ds.size))  # cast to int on insert
                )

    ret = (time_ds, )
    if data_ds != ():
        ret += (data_ds,)
    if indices_ds != ():
        ret += (indices_ds,)

    return ret
