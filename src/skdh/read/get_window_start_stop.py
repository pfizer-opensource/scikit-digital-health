"""
reader utility functions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import insert, append


def get_window_start_stop(indices, n_samples):
    """
    Get the correct starts and stops of windows from indices from the GeneActiv reader

    Parameters
    ----------
    indices : numpy.ndarray
        (N, ) array of indices, where N is the number of pages in the file. If the time
        for that page doesn't overlap with either the base/period of a window, then its
        value is set to `-2*expected_num_samples`.
    n_samples : int
        Number of samples in the data read from the GeneActiv file.

    Returns
    -------
    starts : numpy.ndarray
        (M, ) array of the start indices of windows
    stops : numpy.ndarray
        (M, ) array of the stop indices of windows.
    """
    # reader saves indices corresponding to base hour as positive
    base_mask = indices > 0
    # indices correspondingto base + period hour are saved as negative
    period_mask = (indices < 0) & (indices > -(n_samples + 1))

    # temp indices
    start_ = indices[base_mask]
    stop_ = -indices[period_mask]

    if stop_.size == 0:  # only base indices are saved if period is 24h
        starts = insert(start_, 0, 0)
        stops = append(start_, n_samples)
    elif stop_[0] > start_[0]:  # data starts before the first full day
        starts = start_
        if stop_[-1] > start_[-1]:  # data ends after last full day
            stops = stop_
        else:
            stops = append(stop_, n_samples)
    else:  # data stars in the middle of a window
        starts = insert(start_, 0, 0)
        if stop_[-1] > start_[-1]:
            stops = stop_
        else:
            stops = append(stop_, n_samples)

    return starts, stops
