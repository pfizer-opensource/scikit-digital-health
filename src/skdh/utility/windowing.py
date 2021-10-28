"""
Utility methods

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import require
from numpy.lib.stride_tricks import as_strided

__all__ = ["compute_window_samples", "get_windowed_view"]


class DimensionError(Exception):
    """
    Custom error for if the input signal has too many dimensions
    """

    pass


class ContiguityError(Exception):
    """
    Custom error for if the input signal is not C-contiguous
    """

    pass


def compute_window_samples(fs, window_length, window_step):
    """
    Compute the number of samples for a window. Takes the sampling frequency, window length, and
    window step in common representations and converts them into number of samples.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    window_length : float
        Window length in seconds. If not provided (None), will do no windowing. Default is None
    window_step : {float, int}
        Window step - the spacing between the start of windows. This can be specified several
        different ways (see Notes). Default is 1.0

    Returns
    -------
    length_n : int
        Window length in samples
    step_n : int
        Window step in samples

    Raises
    ------
    ValueError
        If `window_step` is negative, or if `window_step` is a float not in (0.0, 1.0]

    Notes
    -----
    Computation of the window step depends on the type of input provided, and the range.
    - `window_step` is a float in (0.0, 1.0]: specifies the fraction of a window to skip to get to
    the start of the next window
    - `window_step` is an integer > 1: specifies the number of samples to skip to get to the start
    of the next window

    Examples
    --------
    Compute the window length and step in samples for a 3s window with 50% overlap, with a
    sampling rate of 50Hz

    >>> compute_window_samples(50.0, 3.0, 0.5)
    (150, 75)

    Compute the window length for a 4.5s window with a step of 1 sample, and a sampling
    rate of 100Hz

    >>> compute_window_samples(100.0, 4.5, 1)
    (450, 1)
    """
    if window_step is None or window_length is None:
        return None, None

    length_n = int(round(fs * window_length))

    if isinstance(window_step, int):
        if window_step > 0:
            step_n = window_step
        else:
            raise ValueError("window_step cannot be negative")
    elif isinstance(window_step, float):
        if 0.0 < window_step < 1.0:
            step_n = int(round(length_n * window_step))

            step_n = max(min(step_n, length_n), 1)

        elif window_step == 1.0:
            step_n = length_n
        else:
            raise ValueError("float values for window_step must be in (0.0, 1.0]")

    return length_n, step_n


def get_windowed_view(x, window_length, step_size, ensure_c_contiguity=False):
    """
    Return a moving window view over the data

    Parameters
    ----------
    x : numpy.ndarray
        1- or 2-D array of signals to window. Windows occur along the 0 axis.
        Must be C-contiguous.
    window_length : int
        Window length/size.
    step_size : int
        Step/stride size for windows - how many samples to step from window
        center to window center.
    ensure_c_contiguity : bool, optional
        Create a new array with C-contiguity if the passed array is not C-contiguous.
        This *may* result in the memory requirements significantly increasing. Default is False,
        which will raise a ValueError if `x` is not C-contiguous

    Returns
    -------
    x_win : numpy.ndarray
        2- or 3-D array of windows of the original data, of shape (..., L[, ...])
    """
    if not (x.ndim in [1, 2]):
        raise DimensionError("Array cannot have more than 2 dimensions.")

    if ensure_c_contiguity:
        x = require(x, requirements=["C"])
    else:
        if not x.flags["C_CONTIGUOUS"]:
            raise ContiguityError(
                "Input array must be C-contiguous.  See numpy.ascontiguousarray"
            )

    if x.ndim == 1:
        nrows = ((x.size - window_length) // step_size) + 1
        n = x.strides[0]
        return as_strided(
            x, shape=(nrows, window_length), strides=(step_size * n, n), writeable=False
        )

    else:
        k = x.shape[1]
        nrows = ((x.shape[0] - window_length) // step_size) + 1
        n = x.strides[1]

        new_shape = (nrows, window_length, k)
        new_strides = (step_size * k * n, k * n, n)
        return as_strided(x, shape=new_shape, strides=new_strides, writeable=False)
