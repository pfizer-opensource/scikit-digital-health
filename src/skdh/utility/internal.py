"""
Internal utility functions that don't necessarily need to be exposed in the public API

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import (
    asarray,
    array,
    nonzero,
    insert,
    arange,
    interp,
    zeros,
    around,
    diff,
    mean,
    float_,
    int_,
    ndarray,
    concatenate,
    minimum,
    maximum,
    roll,
)
from scipy.signal import cheby1, sosfiltfilt


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

    # check if we should just return empty
    if all(
        [
            i.size == 0 and j.size == 0 and k
            for i, j, k in zip(starts, stops, for_inclusion)
        ]
    ):
        return asarray([], dtype=int), asarray([], dtype=int)

    # get the subset that intersect the day in a roundabout way
    starts_tmp = list(minimum(maximum(i, day_start), day_stop) for i in starts)
    stops_tmp = list(minimum(maximum(i, day_start), day_stop) for i in stops)
    starts_subset, stops_subset = [], []
    for start, stop, fi in zip(starts_tmp, stops_tmp, for_inclusion):
        if start.size == 0 or stop.size == 0:
            continue
        if fi:  # flip everything to being an "exclude" window
            tmp = insert(roll(start, -1), 0, start[0])
            tmp[-1] = day_stop

            tmp_stop = insert(stop, 0, 0)

            starts_subset.append(tmp_stop[tmp_stop != tmp])
            stops_subset.append(tmp[tmp_stop != tmp])
        else:
            starts_subset.append(start[start != stop])
            stops_subset.append(stop[start != stop])

    # get overlap
    all_starts = (
        concatenate(starts_subset) if len(starts_subset) > 0 else asarray(starts_subset)
    )
    all_stops = (
        concatenate(stops_subset) if len(starts_subset) > 0 else asarray(starts_subset)
    )

    valid_starts, valid_stops = [day_start], [day_stop]

    for start, stop in zip(all_starts, all_stops):
        cond1 = [i <= start <= j for i, j in zip(valid_starts, valid_stops)]
        cond2 = [i <= stop <= j for i, j in zip(valid_starts, valid_stops)]

        for i, (c1, c2) in enumerate(zip(cond1, cond2)):
            if c1 and c2:
                valid_starts.insert(
                    i + 1, stop
                )  # valid_starts[i] = [valid_starts[i], stop]
                valid_stops.insert(i, start)  # valid_stops[i] = [start, valid_stops[i]]
            elif c1:
                valid_stops[i] = start
            elif c2:
                valid_starts[i] = stop

    valid_starts = asarray(valid_starts)
    valid_stops = asarray(valid_stops)

    return (
        valid_starts[valid_starts != valid_stops],
        valid_stops[valid_starts != valid_stops],
    )


def apply_downsample(goal_fs, time, data=(), indices=(), aa_filter=True, fs=None):
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
    aa_filter : bool, optional
        Apply an anti-aliasing filter before downsampling. Default is True. This
        is the same filter as used by :py:function:`scipy.signal.decimate`.
        See [1]_ for details.
    fs : {None, float}, optional
        Original sampling frequency in Hz. If `goal_fs` is an integer factor
        of `fs`, every nth sample will be taken, otherwise `np.interp` will be
        used. Leave blank to always use `np.interp`.

    Returns
    -------
    time_ds : numpy.ndarray
        Downsampled time.
    data_ds : tuple, optional
        Downsampled data, if provided.
    indices : tuple, optional
        Downsampled indices, if provided.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Downsampling_(signal_processing)
    """

    def downsample(x, factor, t, t_ds):
        if int(1 / factor) == 1 / factor:
            if x.ndim == 1:
                return (x[:: int(1 / factor)],)
            elif x.ndim == 2:
                return (x[:: int(1 / factor)],)
        else:
            if x.ndim == 1:
                return (interp(t_ds, t, x),)
            elif x.ndim == 2:
                xds = zeros((t_ds.size, x.shape[1]), dtype=float_)
                for j in range(x.shape[1]):
                    xds[:, j] = interp(t_ds, t, x[:, j])
                return (xds,)

    if fs is None:
        # compute the sampling frequency by hand
        fs = 1 / mean(diff(time[:2500]))

    if int(fs / goal_fs) == fs / goal_fs:
        time_ds = time[:: int(fs / goal_fs)]
    else:
        time_ds = arange(time[0], time[-1], 1 / goal_fs)
    # AA filter, if necessary
    sos = cheby1(8, 0.05, 0.8 / (fs / goal_fs), output="sos")

    data_ds = ()

    for dat in data:
        if dat is None:
            data_ds += (None,)
        elif dat.ndim in [1, 2]:
            data_to_ds = sosfiltfilt(sos, dat, axis=0) if aa_filter else dat
            data_ds += downsample(data_to_ds, fs / goal_fs, time, time_ds)
        else:
            raise ValueError("Data dimension exceeds 2, or data not understood.")

    # downsampling indices
    indices_ds = ()
    for idx in indices:
        if idx is None:
            indices_ds += (None,)
        elif idx.ndim == 1:
            indices_ds += (
                around(interp(time[idx], time_ds, arange(time_ds.size))).astype(int_),
            )
        elif idx.ndim == 2:
            indices_ds += (zeros(idx.shape, dtype=int_),)
            for i in range(idx.shape[1]):
                indices_ds[-1][:, i] = around(
                    interp(
                        time[idx[:, i]], time_ds, arange(time_ds.size)
                    )  # cast to int on insert
                )

    ret = (time_ds,)
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


def invert_indices(starts, stops, zero_index, end_index):
    """
    Invert indices from one set of starts and stops, to the opposite

    Parameters
    ----------
    starts : numpy.ndarray
        Array of start indices
    stops : numpy.ndarray
        Array of stop indices
    zero_index : int
        Start index for the array indexing into
    end_index : int
        End index for the array indexing into

    Returns
    -------
    inv_starts : numpy.ndarray
        Inverted array of start indices
    inv_stops : numpy.ndarray
        Inverted array of stop indices
    """
    if starts.size != stops.size:
        raise ValueError("starts and stops indices arrays must be the same size")
    if starts.size == 0:
        return array([zero_index]), array([end_index])

    """
    if fi:  # flip everything to being an "exclude" window
        tmp = insert(roll(start, -1), 0, start[0])
        tmp[-1] = day_stop
    
        tmp_stop = insert(stop, 0, 0)
    
        starts_subset.append(tmp_stop[tmp_stop != tmp])
        stops_subset.append(tmp[tmp_stop != tmp])
    """
    # in general, stops become starts, and starts become stops
    inv_stops = insert(roll(starts, -1), 0, starts[0])
    inv_stops[-1] = end_index

    inv_starts = insert(stops, 0, 0)

    # check for 0 length windows
    mask = inv_starts != inv_stops

    return inv_starts[mask], inv_stops[mask]
