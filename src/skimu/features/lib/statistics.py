"""
Signal features based on statistics measures

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import max, min, quantile, mean, std, arange

from skimu.features.core import Feature
from skimu.features.lib import extensions

__all__ = ['Range', 'IQR', 'RMS', 'Autocorrelation', 'LinearSlope']


class Range(Feature):
    """
    The difference between the maximum and minimum value.
    """
    def __init__(self):
        super().__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the range

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the range for.
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
        range : numpy.ndarray
            Signal range.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return max(x, axis=-1) - min(x, axis=-1)


class IQR(Feature):
    """
    The difference between the 75th percentile and 25th percentile of the values.
    """
    def __init__(self):
        super(IQR, self).__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the IQR

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the IQR for.
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
        iqr : numpy.ndarray
            Signal IQR.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return quantile(x, 0.75, axis=-1) - quantile(x, 0.25, axis=-1)


class RMS(Feature):
    """
    The root mean square value of the signal
    """
    def __init__(self):
        super(RMS, self).__init__('RMS', {})

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the RMS

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the RMS for.
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
        rms : numpy.ndarray
            Signal RMS.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
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
    def __init__(self, lag=1, normalize=True):
        super(Autocorrelation, self).__init__(
            lag=lag,
            normalize=normalize
        )

        self.lag = lag
        self.normalize = normalize

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the autocorrelation

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the autocorrelation for.
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
        ac : numpy.ndarray
            Signal autocorrelation.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.autocorrelation(x, self.lag, self.normalize)


class LinearSlope(Feature):
    """
    The slope from linear regression of the signal
    """
    def __init__(self):
        super(LinearSlope, self).__init__()

    def compute(self, signal, fs, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the linear regression slope

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the linear slope for.
        fs : float
            Sampling frequency in Hz.
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
        slope : numpy.ndarray
            Signal slope.
        """
        return super().compute(signal, fs, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
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
