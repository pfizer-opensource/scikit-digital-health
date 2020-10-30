"""
Utility functions for features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import ndarray, ascontiguousarray
from pandas import DataFrame

from skimu.utility import get_windowed_view


class InputTypeError(Exception):
    """
    Custom exception for the wrong input type
    """
    pass


def standardize_signal(signal, windowed=False, window_length=None, step=None, columns=None):
    """
    Standardize an incoming signal to be consistent across conditions

    Parameters
    ----------
    signal : {numpy.ndarray, pandas.DataFrame}
        Signal to be analyzed. Either ndarray (up to 3d), or a DataFrame.
    windowed : bool, optional
        The signal has already been windowed. Default is False.
    window_length : int, optional
        Window length in samples. If not provided (None), or input data is 3D, input data is
        assumed to be the signal on which to compute features.
    step : int, optional
        Window step in samples. If not provided (None), or input data is 3D, input data is assumed
        to be the signal on which to compute features.
    columns : array-like, optional
        If providing a DataFrame, the columns to use for the signal.

    Returns
    -------
    standard_signal : numpy.ndarray
        3D signal with standard setup
    columns : array-like
        Column names used in signal computation

    Raises
    ------
    DimensionError
        If the signal has more than 3 dimensions
    InputTypeError
        If the signal is not a numpy.ndarray or pandas.DataFrame
    """
    # get a numpy array of the signal
    if isinstance(signal, ndarray):
        if signal.ndim > 3:
            raise DimensionError(f'signal dimensions ({signal.ndim}) exceed dimension max of 3')
        x = ascontiguousarray(signal)
    elif isinstance(signal, DataFrame):
        if columns is not None:
            x = ascontiguousarray(signal[columns].values)
        else:
            x = ascontiguousarray(signal.values)
            columns = signal.columns
    else:
        raise InputTypeError(
            f"signal must be a numpy.ndarray or pandas.DataFrame, not {type(signal)}"
        )

    # determine if going to window the signal
    if (not windowed) and (x.ndim < 3) and (window_length is not None) and (step is not None):
        x = get_windowed_view(x, window_length, step)
        windowed = True

    # standardize to a 3D input
    ret = tuple()  # return value
    if x.ndim == 1:
        ret += (x.reshape((1, -1, 1)), )
    elif x.ndim == 2:
        if windowed:
            ret += (x.reshape(x.shape + (1, )), )
        else:
            ret += (x.reshape((1, ) + x.shape), )
    else:  # x is already 3D
        ret += (x, )

    ret += (columns, )

    return ret
