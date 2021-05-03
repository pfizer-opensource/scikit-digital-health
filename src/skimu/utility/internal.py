"""
Internal utility functions that don't necessarily need to be exposed in the public API

Lukas Adamowicz
Pfizer DMTI 2021
"""
from numpy import asarray, nonzero, insert, append, arange, interp, zeros, around, diff, float_,\
    int_, ndarray, concatenate, minimum, maximum, roll


def get_day_index_intersection(starts, stops, for_inclusion, day_start, day_stop):
    """
    Get the intersection between day start and stop indices and various start and stop indices
    that may or may not happen during the day, and are not necessarily for inclusion.

    Parameters
    ----------
    starts : numpy.ndarray, tuple
        Single ndarray or tuple of ndarrays indicating the starts of events to either include or
        exclude from during the day.
    stops : numpy.ndarray, tuple
    Single ndarray or tuple of ndarrays indicating the stops of events to either include or
        exclude from during the day.
    for_inclusion : bool, tuple
        Single or tuple of booleans indicating if the corresponding start & stop indices are
        for inclusion or not.
    day_start : int
        Day start index.
    day_stop : int
        Day stop index.

    Returns
    -------
    valid_starts : numpy.ndarray
        Intersection of overlapping day and event start indices that are valid/usable.
    valid_stops : numpy.ndarray
        Intersection of overlapping day and event stop indices that are valid/usable.
    """
    day_start, day_stop = int(day_start), int(day_stop)

    # make a common format instead of having to deal with different formats later
    if isinstance(starts, ndarray):
        starts = (starts,)
    if isinstance(stops, ndarray):
        stops = (stops,)

    if len(starts) != len(stops):
        raise ValueError("Number of start arrays does not match number of stop arrays.")
    if isinstance(for_inclusion, bool):
        for_inclusion = (for_inclusion,) * len(starts)

    # get the subset that intersect the day in a roundabout way
    starts_tmp = list(
        minimum(maximum(i, day_start), day_stop) for i in starts
    )
    stops_tmp = list(
        minimum(maximum(i, day_start), day_stop) for i in stops
    )
    starts_subset, stops_subset = [], []
    for start, stop, fi in zip(starts_tmp, stops_tmp, for_inclusion):
        if fi:  # flip everything to being an "exclude" window
            tmp = roll(start, -1)
            tmp[-1] = day_stop

            starts_subset.append(stop[stop != tmp])
            stops_subset.append(tmp[stop != tmp])
        else:
            starts_subset.append(start[start != stop])
            stops_subset.append(stop[start != stop])

    # get overlap
    all_starts = concatenate(starts_subset)
    all_stops = concatenate(stops_subset)

    valid_starts, valid_stops = [day_start], [day_stop]

    for start, stop in zip(all_starts, all_stops):
        cond1 = [i <= start <= j for i, j in zip(valid_starts, valid_stops)]
        cond2 = [i <= stop <= j for i, j in zip(valid_starts, valid_stops)]

        for i, (c1, c2) in enumerate(zip(cond1, cond2)):
            if c1 and c2:
                valid_starts.insert(i + 1, stop)  # valid_starts[i] = [valid_starts[i], stop]
                valid_stops.insert(i, start)  # valid_stops[i] = [start, valid_stops[i]]
            elif c1:
                valid_stops[i] = start
            elif c2:
                valid_starts[i] = stop

    valid_starts = asarray(valid_starts)
    valid_stops = asarray(valid_stops)

    return valid_starts[valid_starts != valid_stops], valid_stops[valid_starts != valid_stops]


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
                return asarray([day_start]), asarray([day_stop])
            else:
                return asarray([]), asarray([])
        except IndexError:
            return asarray([]), asarray([])
    if starts_subset.size == 0 and stops_subset.size == 1:
        starts_subset = asarray([day_start])
    if starts_subset.size == 1 and stops_subset.size == 0:
        stops_subset = asarray([day_stop])

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
            data_ds += (None,)
        elif dat.ndim == 1:
            data_ds += (interp(time_ds, time, dat),)
        elif dat.ndim == 2:
            data_ds += (zeros((time_ds.size, dat.shape[1]), dtype=float_),)
            for i in range(dat.shape[1]):
                data_ds[-1][:, i] = interp(time_ds, time, dat[:, i])
        else:
            raise ValueError("Data dimension exceeds 2, or data not understood.")

    # downsampling indices
    indices_ds = ()
    for idx in indices:
        if idx is None:
            indices_ds += (None,)
        elif idx.ndim == 1:
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


def rle(to_encode):
    """
    Run length encoding.

    Parameters
    ----------
    to_encode : array-like

    Returns
    -------
    lengths : array
        Lengths of each block.
    block_start_indices : array
        Indices of the start of each block.
    block_values : array
        The value repeated for the duration of each block.
    """
    starts = nonzero(diff(to_encode))[0] + 1
    # add the end too for length computation
    starts = insert(starts, (0, starts.size), (0, len(to_encode)))

    lengths = diff(starts)
    starts = starts[:-1]  # remove that last index which isn't actually a start
    values = asarray(to_encode)[starts]

    return lengths, starts, values
