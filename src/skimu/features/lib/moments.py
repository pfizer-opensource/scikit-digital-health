"""
Features from statistical moments

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import mean, std, sum, diff, sign
from scipy.stats import skew, kurtosis

from skimu.features.core import Feature


__all__ = ['Mean', 'MeanCrossRate', 'StdDev', 'Skewness', 'Kurtosis']


class Mean(Feature):
    """
    The signal mean.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.arange(15).reshape((5, 3))
    >>> mn = Mean()
    >>> mn.compute(signal)
    array([6., 7., 8.])
    """
    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the mean.

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the mean for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        mean : numpy.ndarray
            Computed mean.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return mean(x, axis=-1)


class MeanCrossRate(Feature):
    """
    Number of signal mean value crossings. Expressed as a percentage of signal length.
    """
    def __init__(self):
        super(MeanCrossRate, self).__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the mean cross rate

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the mean cross rate for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        mcr : numpy.ndarray
            Computed mean cross rate.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        x_nomean = x - mean(x, axis=-1, keepdims=True)
        mcr = sum(diff(sign(x_nomean), axis=-1) != 0, axis=-1)

        return mcr / x.shape[-1]  # shape of the 1 axis


class StdDev(Feature):
    """
    The signal standard deviation

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.arange(15).reshape((5, 3))
    >>> StdDev().compute(signal)
    array([[4.74341649, 4.74341649, 4.74341649]])
    """
    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the standard deviation

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the standard deviation for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        stdev : numpy.ndarray
            Computed standard deviation.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return std(x, axis=-1, ddof=1)


class Skewness(Feature):
    """
    The skewness of a signal. NaN inputs will be propagated through to the result.
    """
    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the skewness

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the skewness for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        skew : numpy.ndarray
            Computed skewness.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return skew(x, axis=-1, bias=False)


class Kurtosis(Feature):
    """
    The kurtosis of a signal. NaN inputs will be propagated through to the result.
    """
    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the kurtosis

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the kurtosis for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        kurt : numpy.ndarray
            Computed kurtosis.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return kurtosis(x, axis=-1, bias=False)
