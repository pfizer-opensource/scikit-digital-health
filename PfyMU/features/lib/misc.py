"""
Misc features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


__all__ = ['ComplexityInvariantDistance', 'RangeCountPercentage', 'RatioBeyondRSigma']


class ComplexityInvariantDistance(Feature):
    def __init__(self, normalize=True):
        """
        Compute a measure of distance in a signal that is invariant to its complexity

        Parameters
        ----------
        normalize : bool, optional
            Normalize the signal. Default is True.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(ComplexityInvariantDistance, self).__init__('ComplexityInvariantDistance', {'Normalize': normalize})
        self.normalize = normalize

    def _compute(self, x, fs):
        super(ComplexityInvariantDistance, self)._compute(x, fs)
        self._result = _cython.CID(x, self.normalize)


class RangeCountPercentage(Feature):
    def __init__(self, range_min=-1.0, range_max=1.0):
        """
        Compute the percent of the signal that falls between the minimum and maximum values

        Parameters
        ----------
        range_min : {int, float}, optional
            Minimum value of the range. Default value is -1.0
        range_max : {int, float}, optional
            Maximum value of the range. Default value is 1.0

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(RangeCountPercentage, self).__init__(
            'RangeCountPercentage',
            {'Range min': range_min, 'Range max': range_max}
        )

        self.rmin = range_min
        self.rmax = range_max

    def _compute(self, x, fs):
        super(RangeCountPercentage, self)._compute(x, fs)

        self._result = _cython.RangeCount(x, self.rmin, self.rmax)


class RatioBeyondRSigma(Feature):
    def __init__(self, r=2.0):
        """
        Compute the percent of the signal that is farther than :math:`r\sigma(x)` away from the mean of the signal.

        Parameters
        ----------
        r : float, optional
            Number of standard deviations above or below the mean the range includes. Default is 2.0

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(RatioBeyondRSigma, self).__init__(
            'RatioBeyondRSigma',
            {'r': r}
        )

        self.r = r

    def _compute(self, x, fs):
        super(RatioBeyondRSigma, self)._compute(x, fs)

        self._result = _cython.RatioBeyondRSigma(x, self.r)
