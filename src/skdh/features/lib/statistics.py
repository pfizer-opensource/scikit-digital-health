"""
Signal features based on statistics measures

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from numpy import max, min, quantile, mean, std

from skdh.features.core import Feature
from skdh.features.lib import extensions

__all__ = ["Range", "IQR", "RMS", "Autocorrelation", "LinearSlope"]


class Range(Feature):
    """
    The difference between the maximum and minimum value.
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the range

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the range for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        range : numpy.ndarray
            Signal range.
        """
        x = super().compute(signal, axis=axis)
        return max(x, axis=-1) - min(x, axis=-1)


class IQR(Feature):
    """
    The difference between the 75th percentile and 25th percentile of the values.
    """

    __slots__ = ()

    def __init__(self):
        super(IQR, self).__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the IQR

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the IQR for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        iqr : numpy.ndarray
            Signal IQR.
        """
        x = super().compute(signal, axis=axis)
        return quantile(x, 0.75, axis=-1) - quantile(x, 0.25, axis=-1)


class RMS(Feature):
    """
    The root mean square value of the signal
    """

    __slots__ = ()

    def __init__(self):
        super(RMS, self).__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the RMS

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the RMS for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        rms : numpy.ndarray
            Signal RMS.
        """
        x = super().compute(signal, axis=axis)
        return std(x - mean(x, axis=-1, keepdims=True), axis=-1, ddof=1)


class Autocorrelation(Feature):
    """
    The similarity in profile between the signal and a time shifted version of the signal.

    Parameters
    ----------
    lag : int, optional
        Amount of lag (in samples) to use for the autocorrelation. Default is 1 sample.
    normalize : bool, optional
        Normalize the result using the mean/std. deviation. Default is True
    """

    __slots__ = ("lag", "normalize")

    def __init__(self, lag=1, normalize=True):
        super(Autocorrelation, self).__init__(lag=lag, normalize=normalize)

        self.lag = lag
        self.normalize = normalize

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the autocorrelation

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the autocorrelation for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        ac : numpy.ndarray
            Signal autocorrelation.
        """
        x = super().compute(signal, axis=axis)
        return extensions.autocorrelation(x, self.lag, self.normalize)


class LinearSlope(Feature):
    """
    The slope from linear regression of the signal
    """

    __slots__ = ()

    def __init__(self):
        super(LinearSlope, self).__init__()

    def compute(self, signal, fs=1.0, *, axis=-1):
        """
        Compute the linear regression slope

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the linear slope for.
        fs : float, optional
            Sampling frequency in Hz. If not provided, default is 1.0Hz.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        slope : numpy.ndarray
            Signal slope.
        """
        x = super().compute(signal, fs, axis=axis)
        return extensions.linear_regression(x, fs)


'''
# TODO implement
class AutoregressiveCoefficients(Feature):
    def __init__(self):
        """
        Compute the specified autoregressive coefficient

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(AutoregressiveCoefficients, self).__init__('AutoregressiveCoefficients', {})

        raise NotImplementedError('Feature not yet implemented')


# TODO implement
class AutocovarianceIQR(Feature):
    def __init__(self):
        """
        Compute the inter-Quartile range of the autocovariance

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(AutocovarianceIQR, self).__init__('AutocovarianceIQR', {})

        raise NotImplementedError('Feature not yet implemented')
'''
