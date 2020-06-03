"""
Utility functions for features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import ndarray, ascontiguousarray, round
from numpy.lib.stride_tricks import as_strided
from pandas import DataFrame

__all__ = ['compute_window_samples', 'get_windowed_view']


class InputTypeError(Exception):
    """
    Custom exception for the wrong input type
    """
    pass


class DimensionError(Exception):
    """
    Custom error for if the input signal has too many dimensions
    """
    pass


def standardize_signal(signal, window_length=None, step=None, columns=None):
    """
    Standardize an incoming signal to be consistent across conditions

    Parameters
    ----------
    signal : {numpy.ndarray, pandas.DataFrame}
        Signal to be analyzed. Either ndarray (up to 3d), or a DataFrame.
    window_length : int, optional
        Window length in samples. If not provided (None), or input data is 3D, input data is assumed to be the signal on
        which to compute features.
    step : int, optional
        Window step in samples. If not provided (None), or input data is 3D, input data is assumed to be the signal on
        which to compute features.
    columns : array-like, optional
        If providing a DataFrame, the columns to use for the signal.

    Returns
    -------
    standard_signal : numpy.ndarray
        3D signal with standard setup
    columns : array-like
        Column names used in signal computation
    """
    # get a numpy array of the signal
    if isinstance(signal, ndarray):
        if signal.ndim > 3:
            raise DimensionError(f'signal dimensions ({signal.ndim}) exceeds dimension maximum of 3')
        x = ascontiguousarray(signal)
    elif isinstance(signal, DataFrame):
        if columns is not None:
            x = ascontiguousarray(signal[columns].values)
        else:
            x = ascontiguousarray(signal.values)
            columns = signal.columns
    else:
        raise InputTypeError(f"signal must be a numpy.ndarray or pandas.DataFrame, not {type(signal)}")

    # determine if going to window the signal
    windowed = False  # keep track of if windowing occurred
    if x.ndim < 3 and window_length is not None and step is not None:
        x = get_windowed_view(x, window_length, step)
        windowed = True

    # standardize to a 3D input
    ret = tuple()
    if x.ndim == 1:
        ret += (x.reshape((1, -1, 1)), )
    elif x.ndim == 2:
        if windowed:
            ret += (x.reshape(x.shape + (1, )), )
        else:
            ret += (x.reshape((-1, ) + x.shape), )
    else:  # x is already 3D
        ret += (x, )

    if isinstance(signal, DataFrame):
        ret += (columns, )

    return ret


def compute_window_samples(fs, window_length, window_step):
    """

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    window_length : float
        Window length in seconds. If not provided (None), will do no windowing. Default is None
    window_step : {float, int}
        Window step - the spacing between the start of windows. This can be specified several different ways
        (see Notes). Default is 1.0

    Returns
    -------
    length_n : int
        Window length in samples
    step_n : int
        Window step in samples

    Notes
    -----
    Computation of the window step depends on the type of input provided, and the range.
    - `window_step` is a float in (0.0, 1.0]: specifies the fraction of a window to skip to get to the start of the
    next window
    - `window_step` is an integer > 1: specifies the number of samples to skip to get to the start of the next
    window

    Examples
    --------
    Compute the window length and step in samples for a 3s window with 50% overlap, with a sampling rate of 50Hz

    >>> compute_window_samples(50.0, 3.0, 0.5)
    (150, 75)

    Compute the window length for a 4.5s window with a step of 1 sample, and a sampling rate of 100Hz
    >>> compute_window_samples(100.0, 4.5, 1)
    (450, 1)
    """
    length_n = int(round(fs * window_length))

    if isinstance(window_step, int):
        if window_step > 0:
            step_n = window_step
        else:
            raise ValueError("window_step cannot be negative")
    elif isinstance(window_step, float):
        if 0.0 < window_step < 1.0:
            step_n = int(round(length_n * window_step))

            if step_n < 1:
                step_n = 1
            if step_n > length_n:
                step_n = length_n

        elif window_step == 1.0:
            step_n = length_n
        else:
            raise ValueError("float values for window_step must be in (0.0, 1.0]")

    return length_n, step_n


def get_windowed_view(x, win_len, stepsize):
    """
    Return a moving window view over the data.

    Parameters
    ----------
    x : numpy.ndarray
        1- or 2-D array of signals to window. Windows occur along the 0 axis. MUST BE C-CONTIGUOUS.
    win_len : int
        Window length.
    stepsize : int
        Stride length/step size. How many places to step for the center of the windows being created.

    Returns
    -------
    x_win : numpy.ndarray
        2D array of windows of the original data, with shape (-1, L)
    """
    if not (x.ndim in [1, 2]):
        raise ValueError('Array cannot have more than 2 dimensions to window properly.')
    if not x.flags['C_CONTIGUOUS']:
        raise ValueError('Array must be C-contiguous to window properly.')
    if x.ndim == 1:
        nrows = ((x.size - win_len) // stepsize) + 1
        n = x.strides[0]
        return as_strided(x, shape=(nrows, win_len), strides=(stepsize * n, n), writeable=False)
    else:
        k = x.shape[1]
        nrows = ((x.shape[0] - win_len) // stepsize) + 1
        n = x.strides[1]

        new_shape = (nrows, win_len, k)
        new_strides = (stepsize * k * n, k * n, n)
        return as_strided(x, shape=new_shape, strides=new_strides, writeable=False)
