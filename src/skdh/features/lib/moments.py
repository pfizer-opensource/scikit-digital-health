"""
Features from statistical moments

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import mean, std, sum, diff, sign
from scipy.stats import skew, kurtosis

from skdh.features.core import Feature


__all__ = ["Mean", "MeanCrossRate", "StdDev", "Skewness", "Kurtosis"]


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

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the mean.

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the mean for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        mean : numpy.ndarray
            Computed mean.
        """
        x = super().compute(signal, axis=axis)
        return mean(x, axis=-1)


class MeanCrossRate(Feature):
    """
    Number of signal mean value crossings. Expressed as a percentage of signal length.
    """

    __slots__ = ()

    def __init__(self):
        super(MeanCrossRate, self).__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the mean cross rate

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the mean cross rate for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        mcr : numpy.ndarray
            Computed mean cross rate.
        """
        x = super().compute(signal, axis=axis)

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

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the standard deviation

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the standard deviation for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        stdev : numpy.ndarray
            Computed standard deviation.
        """
        x = super().compute(signal, axis=axis)
        return std(x, axis=-1, ddof=1)


class Skewness(Feature):
    """
    The skewness of a signal. NaN inputs will be propagated through to the result.
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the skewness

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the skewness for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        skew : numpy.ndarray
            Computed skewness.
        """
        x = super().compute(signal, axis=axis)
        return skew(x, axis=-1, bias=False)


class Kurtosis(Feature):
    """
    The kurtosis of a signal. NaN inputs will be propagated through to the result.
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the kurtosis

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the kurtosis for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        kurt : numpy.ndarray
            Computed kurtosis.
        """
        x = super().compute(signal, axis=axis)
        return kurtosis(x, axis=-1, bias=False)
