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

    Methods
    -------
    compute(signal[, columns=None, windowed=False])

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

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = mean(x, axis=1)


class MeanCrossRate(Feature):
    """
    Number of signal mean value crossings. Expressed as a percentage of signal length.

    Methods
    -------
    compute(signal[, columns=None, windowed=False])
    """
    def __init__(self):
        super(MeanCrossRate, self).__init__('MeanCrossRate', {})

    def _compute(self, x, fs):
        super(MeanCrossRate, self)._compute(x, fs)

        x_nomean = x - mean(x, axis=1, keepdims=True)
        mcr = sum(diff(sign(x_nomean), axis=1) != 0, axis=1)

        self._result = mcr / x.shape[1]  # shape of the 1 axis


class StdDev(Feature):
    """
    The signal standard deviation

    Methods
    -------
    compute(signal[, columns=None, windowed=False])

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.arange(15).reshape((5, 3))
    >>> StDev().compute(signal)
    array([[4.74341649, 4.74341649, 4.74341649]])
    """
    def __init__(self):
        super().__init__('StdDev', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = std(x, axis=1, ddof=1)


class Skewness(Feature):
    """
    The skewness of a signal. NaN inputs will be propagated through to the result.

    Methods
    -------
    compute(signal[, columns=None, windowed=False])
    """
    def __init__(self):
        super().__init__('Skewness', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = skew(x, axis=1, bias=False)


class Kurtosis(Feature):
    """
    The kurtosis of a signal. NaN inputs will be propagated through to the result.

    Methods
    -------
    compute(signal[, columns=None, windowed=False])
    """
    def __init__(self):
        super().__init__('Kurtosis', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = kurtosis(x, axis=1, bias=False)
