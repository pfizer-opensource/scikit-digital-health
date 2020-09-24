"""
Features from statistical moments

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import mean, std, sum, diff, sign
from scipy.stats import skew, kurtosis

from src.PfyMU.features import Feature


__all__ = ['Mean', 'MeanCrossRate', 'StdDev', 'Skewness', 'Kurtosis']


class Mean(Feature):
    def __init__(self):
        """
        Compute the signal mean.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])

        Examples
        --------
        >>> signal = np.arange(15).reshape((5, 3))
        >>> mn = features.Mean()
        >>> mn.compute(signal)
        array([6., 7., 8.])
        """
        super().__init__('Mean', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = mean(x, axis=1)


class MeanCrossRate(Feature):
    def __init__(self):
        """
        Calculate the number of times the signal crosses the mean value, expressed as a percentage of the signal
        length in samples.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(MeanCrossRate, self).__init__('MeanCrossRate', {})

    def _compute(self, x, fs):
        super(MeanCrossRate, self)._compute(x, fs)

        x_nomean = x - mean(x, axis=1, keepdims=True)
        mcr = sum(diff(sign(x_nomean), axis=1) != 0, axis=1)

        self._result = mcr / x.shape[1]  # shape of the 1 axis


class StdDev(Feature):
    def __init__(self):
        """
        Compute the signal standard deviation.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])

        Examples
        --------
        >>> signal = np.arange(15).reshape((5, 3))
        >>> features.StDev().compute(signal)
        array([[4.74341649, 4.74341649, 4.74341649]])
        """
        super().__init__('StdDev', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = std(x, axis=1, ddof=1)


class Skewness(Feature):
    def __init__(self):
        """
        Compute the skewness of a signal. NaN inputs will be propagated through to the result.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super().__init__('Skewness', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = skew(x, axis=1, bias=False)


class Kurtosis(Feature):
    def __init__(self):
        """
        Compute the kurtosis of a signal. NaN inputs will be propagated through to the result.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super().__init__('Kurtosis', {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = kurtosis(x, axis=1, bias=False)
