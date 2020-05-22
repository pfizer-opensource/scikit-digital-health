"""
Utility functions for features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import ndarray
from pandas import DataFrame

from PfyMU.features.core import InputTypeError


def standardize_signal(signal, window=None, step=None, columns=None):
    """
    Standardize an incoming signal to be consistent across conditions

    Parameters
    ----------
    signal : {numpy.ndarray, pandas.DataFrame}
        Signal to be analyzed. Either ndarray (up to 3d), or a DataFrame.
    window : int, optional
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
        x = signal
    elif isinstance(signal, DataFrame):
        if columns is not None:
            x = signal[columns].values
        else:
            x = signal.values
            columns = signal.columns
    else:
        raise InputTypeError(f"signal must be a numpy.ndarray or pandas.DataFrame, not {type(signal)}")

    # determine if going to window the signal
