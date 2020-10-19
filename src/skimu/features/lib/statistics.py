"""
Signal features based on statistics measures

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import max, min, quantile, mean, std, arange

from skimu.features.core import Feature
from skimu.features.lib import _cython

__all__ = ['Range', 'IQR', 'RMS', 'Autocorrelation', 'LinearSlope']


class Range(Feature):
    """
    The difference between the maximum and minimum value.
    """
    def __init__(self):
        super().__init__("Range", {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the range

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        range : {numpy.ndarray, pandas.DataFrame}
            Signal range, returned as the same type as `signal`
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = max(x, axis=1) - min(x, axis=1)


class IQR(Feature):
    """
    The difference between the 75th percentile and 25th percentile of the values.
    """
    def __init__(self):
        super(IQR, self).__init__("IQR", {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the IQR

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        iqr : {numpy.ndarray, pandas.DataFrame}
            Signal IQR, returned as the same type as `signal`
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(IQR, self)._compute(x, fs)

        self._result = quantile(x, 0.75, axis=1) - quantile(x, 0.25, axis=1)


class RMS(Feature):
    """
    The root mean square value of the signal

    Methods
    -------
    compute(signal[, columns=None, windowed=False])
    """
    def __init__(self):
        super(RMS, self).__init__('RMS', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the RMS

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        rms : {numpy.ndarray, pandas.DataFrame}
            Signal RMS, returned as the same type as `signal`
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(RMS, self)._compute(x, fs)

        self._result = std(x - mean(x, axis=1, keepdims=True), axis=1, ddof=1)


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
            'Autocorrelation', {'lag': lag, 'normalize': normalize}
        )

        self.lag = lag
        self.normalize = normalize

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the autocorrelation

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        ac : {numpy.ndarray, pandas.DataFrame}
            Signal autocorrelation, returned as the same type as `signal`
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(Autocorrelation, self)._compute(x, fs)

        self._result = _cython.Autocorrelation(x, self.lag, self.normalize)


class LinearSlope(Feature):
    """
    The slope from linear regression of the signal
    """
    def __init__(self):
        super(LinearSlope, self).__init__('LinearSlope', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the linear regression slope

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        slope : {numpy.ndarray, pandas.DataFrame}
            Signal slope, returned as the same type as `signal`
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(LinearSlope, self)._compute(x, fs)

        t = arange(x.shape[1]) / fs

        self._result, intercept = _cython.LinRegression(t, x)


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
