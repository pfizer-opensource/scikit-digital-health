"""
Utility functions for features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import ndarray, ascontiguousarray
from numpy.lib.stride_tricks import as_strided
from pandas import DataFrame

from PfyMU.features.core import InputTypeError


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
    if x.ndim == 1:
        return x.reshape((1, -1, 1)), columns
    elif x.ndim == 2:
        if windowed:
            return x.reshape(x.shape + (-1, ))
        else:
            return x.reshape((-1, ) + x.shape)
    else:  # x is already 3D
        return x


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
