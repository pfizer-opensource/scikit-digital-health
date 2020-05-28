"""
Signal features based on statistics measures
"""
from numpy import max, min, quantile, mean, std, arange

from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


class Range(Feature):
    def __init__(self):
        """
        Compute the difference between the maximum and minimum value in a signal

        Methods
        -------
        compute(signal[, columns=None])
        """
        super().__init__("Range", {})

    def _compute(self, x, fs):
        super()._compute(x, fs)

        self._result = max(x, axis=1) - min(x, axis=1)


class IQR(Feature):
    def __init__(self):
        """
        Compute the difference between the 75th percentile and 25th percentile of the values in a signal.

        Methods
        -------
        compute(signal[, columns=None])
        """
        super(IQR, self).__init__("IQR", {})

    def _compute(self, x, fs):
        super(IQR, self)._compute(x, fs)

        self._result = quantile(x, 0.75, axis=1) - quantile(x, 0.25, axis=1)


class RMS(Feature):
    def __init__(self):
        """
        Compute the Root Mean Square value of the signal

        Methods
        -------
        compute(signal[, columns=None])
        """
        super(RMS, self).__init__('RMS', {})

    def _compute(self, x, fs):
        super(RMS, self)._compute(x, fs)

        self._result = std(x - mean(x, axis=1, keepdims=True), axis=1, ddof=1)


class LinearSlope(Feature):
    def __init__(self):
        """
        Compute the linear slope for the signal

        Methods
        -------
        compute(signal[, columns=None])
        """
        super(LinearSlope, self).__init__('LinearSlope', {})

    def _compute(self, x, fs):
        super(LinearSlope, self)._compute(x, fs)

        t = arange(x.shape[1]) / fs

        self._result, intercept = _cython.LinRegression(t, x)


