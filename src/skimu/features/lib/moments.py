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
        super().__init__('Mean', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the mean

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
        mean : {numpy.ndarray, pandas.DataFrame}
            Computed mean, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)
    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = mean(x, axis=1)


class MeanCrossRate(Feature):
    """
    Number of signal mean value crossings. Expressed as a percentage of signal length.
    """
    def __init__(self):
        super(MeanCrossRate, self).__init__('MeanCrossRate', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal *, columns=None, windowed=False)

        Compute the mean cross rate

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        fs : float, optional
            Sampling frequency in Hz
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        mcr : {numpy.ndarray, pandas.DataFrame}
            Computed mean cross rate, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(MeanCrossRate, self)._compute(x, fs)

        x_nomean = x - mean(x, axis=1, keepdims=True)
        mcr = sum(diff(sign(x_nomean), axis=1) != 0, axis=1)

        self._result = mcr / x.shape[1]  # shape of the 1 axis


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
        super().__init__('StdDev', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the standard deviation

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
        stdev : {numpy.ndarray, pandas.DataFrame}
            Computed standard deviation, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = std(x, axis=1, ddof=1)


class Skewness(Feature):
    """
    The skewness of a signal. NaN inputs will be propagated through to the result.
    """
    def __init__(self):
        super().__init__('Skewness', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the skewness

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
        skew : {numpy.ndarray, pandas.DataFrame}
            Computed skewness, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = skew(x, axis=1, bias=False)


class Kurtosis(Feature):
    """
    The kurtosis of a signal. NaN inputs will be propagated through to the result.
    """
    def __init__(self):
        super().__init__('Kurtosis', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the kurtosis

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
        kurt : {numpy.ndarray, pandas.DataFrame}
            Computed kurtosis, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = kurtosis(x, axis=1, bias=False)
