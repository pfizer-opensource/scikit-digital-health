"""
Internal utility functions that don't necessarily need to be exposed in the public API

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from numpy import (
    all,
    asarray,
    argsort,
    array,
    clip,
    nonzero,
    insert,
    append,
    arange,
    interp,
    zeros,
    around,
    diff,
    mean,
    float64,
    int_,
    ndarray,
    concatenate,
    roll,
    full,
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
    starts_subset, stops_subset = [], []
    for start, stop, fi in zip(starts, stops, for_inclusion):
        if start.size == 0 or stop.size == 0:
            continue
        if fi:  # flip everything to being an "exclude" window
            # # 1. sort based on the ends
            # i_e = argsort(stop)
            # start = start[i_e]
            # stop = stop[i_e]
            #
            # # 2. create the temporary ends by appending day_stop to the starts
            # tmp_stop = append(start, day_stop)
            #
            # # 3. sort based on the starts
            # i_s = argsort(start)
            # stop = stop[i_s]
            #
            # # 4. create the temp starts by inserting day_start to the stops
            # tmp_start = insert(stop, 0, day_start)

            # 1. sort based on starts
            i_sort = argsort(start)
            start = start[i_sort]
            stop = stop[i_sort]

            # clip to day starts and stops
            start = clip(start, day_start, day_stop)
            stop = clip(stop, day_start, day_stop)

            # 2. check that stops are increasing (ie sorted as well)
            if not all(stop[1:] >= stop[:-1]):
                raise NotImplementedError(
                    "Window ends are not monotonically increasing after sorting by window starts. "
                    "This behavior is not currently supported."
                )

            tmp_stop = append(start, day_stop)
            tmp_start = insert(stop, 0, day_start)

            starts_subset.append(tmp_start[tmp_stop != tmp_start])
            stops_subset.append(tmp_stop[tmp_stop != tmp_start])
        else:
            # clip to day starts and stops
            start = clip(start, day_start, day_stop)
            stop = clip(stop, day_start, day_stop)

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


def apply_resample(
    *, time, goal_fs=None, time_rs=None, data=(), indices=(), aa_filter=True, fs=None
):
    """
    Apply a re-sample to a set of data.

    Parameters
    ----------
    time : numpy.ndarray
        Array of original timestamps.
    goal_fs : float, optional
        Desired sampling frequency in Hz.  One of `goal_fs` or `time_rs` must be
        provided.
    time_rs : numpy.ndarray, optional
        Resampled time series to sample to. One of `goal_fs` or `time_rs` must be
        provided.
    data : tuple, optional
        Tuple of arrays to normally downsample using interpolation. Must match the
        size of `time`. Can handle `None` inputs, and will return an array of zeros
        matching the downsampled size.
    indices : tuple, optional
        Tuple of arrays of indices to downsample.
    aa_filter : bool, optional
        Apply an anti-aliasing filter before downsampling. Default is True. This
        is the same filter as used by :py:function:`scipy.signal.decimate`.
        See [1]_ for details. Ignored if upsampling.
    fs : {None, float}, optional
        Original sampling frequency in Hz. If `goal_fs` is an integer factor
        of `fs`, every nth sample will be taken, otherwise `np.interp` will be
        used. Leave blank to always use `np.interp`.

    Returns
    -------
    time_rs : numpy.ndarray
        Resampled time.
    data_rs : tuple, optional
        Resampled data, if provided.
    indices_rs : tuple, optional
        Resampled indices, if provided.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Downsampling_(signal_processing)
    """

    def resample(x, factor, t, t_rs):
        if (int(factor) == factor) and (factor > 1):
            # in case that t_rs is provided and ends earlier than t
            n = nonzero(t <= t_rs[-1])[0][-1] + 1
            return (x[: n : int(factor)],)
        else:
            if x.ndim == 1:
                return (interp(t_rs, t, x),)
            elif x.ndim == 2:
                xrs = zeros((t_rs.size, x.shape[1]), dtype=float64)
                for j in range(x.shape[1]):
                    xrs[:, j] = interp(t_rs, t, x[:, j])
                return (xrs,)

    if fs is None:
        # compute sampling frequency by hand
        fs = 1 / mean(diff(time[:5000]))

    if time_rs is None and goal_fs is None:
        raise ValueError("One of `time_rs` or `goal_fs` is required.")

    # get resampled time if necessary
    if time_rs is None:
        if int(fs / goal_fs) == fs / goal_fs and goal_fs < fs:
            time_rs = time[:: int(fs / goal_fs)]
        else:
            time_rs = arange(time[0], time[-1], 1 / goal_fs)
    else:
        goal_fs = 1 / mean(diff(time_rs[:5000]))
        # prevent t_rs from extrapolating
        time_rs = time_rs[time_rs <= time[-1]]

    # AA filter, if necessary
    if (fs / goal_fs) >= 1.0:
        sos = cheby1(8, 0.05, 0.8 / (fs / goal_fs), output="sos")
    else:
        aa_filter = False

    # resample data
    data_rs = ()

    for dat in data:
        if dat is None:
            data_rs += (None,)
        elif dat.ndim in [1, 2]:
            data_to_rs = sosfiltfilt(sos, dat, axis=0) if aa_filter else dat
            data_rs += resample(data_to_rs, fs / goal_fs, time, time_rs)
        else:
            raise ValueError("Data dimension exceeds 2, or data not understood.")

    # resampling indices
    indices_rs = ()
    for idx in indices:
        if idx is None:
            indices_rs += (None,)
        elif idx.ndim == 1:
            indices_rs += (
                around(interp(time[idx], time_rs, arange(time_rs.size))).astype(int_),
            )
        elif idx.ndim == 2:
            indices_rs += (zeros(idx.shape, dtype=int_),)
            for i in range(idx.shape[1]):
                indices_rs[-1][:, i] = around(
                    interp(
                        time[idx[:, i]], time_rs, arange(time_rs.size)
                    )  # cast to in on insert
                )

    ret = (time_rs,)
    if data_rs != ():
        ret += (data_rs,)
    if indices_rs != ():
        ret += (indices_rs,)

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


def fill_data_gaps(time, fs, fill_info, **kwargs):
    """
    Fill gaps in data-streams

    Parameters
    ----------
    time : numpy.ndarray
        Array of unix timestamps, in seconds.
    fs : float
        Sampling frequency. In number of samples in a second.
    fill_info : dict
        Dictionary with keys matching `**kwargs` to specify what value to use
        for filling gaps. If a value is not provided, the gap will be filled with
        zeros.
    **kwargs : dict
        Arrays of data to fill gaps in specified by keyword. Must be the same size
        as `time`. Returned as one dictionary for all arrays specified this way.

    Returns
    -------
    time_filled : numpy.ndarray
        Gap filled time array.
    data_filled : dict
        Dictionary of gap filled data arrays.

    Examples
    --------
    >>> time = np.concatenate((arange(0, 4, 0.01), arange(6, 10, 0.01))
    >>> accel = np.random.default_rng().normal(0, 1, (time.size, 3))
    >>> accel[:, 2] += 1  # add gravity acceleration
    >>> temp = np.random.default_rng().normal(28, 1, time.size)
    >>> fill_info = {"accel": [0, 0, 1]}  # only specifiying fill value for accel
    >>> data_rs = fill_data_gaps(time, 100.0, fill_info, accel=accel, temp=temp)
    >>> print(data_rs.keys())
    dict_keys(['accel', 'temp'])
    """
    # get the first location of gaps in the data - add 1 so that the index reflects
    # the first value AFTER the gap
    gaps = nonzero(diff(time) > (1.5 / fs))[0] + 1
    if gaps.size == 0:
        return time, kwargs

    # create sequences of data with no gaps
    seqs = zeros((gaps.size + 1, 2), dtype=int_)
    seqs[1:, 0] = gaps
    seqs[:-1, 1] = gaps
    seqs[-1, 1] = time.size

    # round-about way, but need to prevent start>>>>>>>>>step
    time_rs = arange(0, (time[-1] - time[0]) + 0.5 / fs, 1 / fs) + time[0]

    # iterate over the datastreams
    ret = {}
    for name, data in kwargs.items():
        shape = list(data.shape)
        shape[0] = time_rs.size
        new_data = full(shape, fill_info.get(name, 0.0), dtype=data.dtype)

        for i1, i2 in seqs[::-1]:
            # get the number of samples offset in the resampled time
            i_offset = int((time[i1] - time_rs[i1]) * fs)

            new_data[i1 + i_offset : i2 + i_offset] = data[i1:i2]

        ret[name] = new_data

    return time_rs, ret
