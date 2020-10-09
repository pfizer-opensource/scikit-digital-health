"""
Misc features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from skimu.features.core import Feature
from skimu.features.lib import _cython

__all__ = ['ComplexityInvariantDistance', 'RangeCountPercentage', 'RatioBeyondRSigma']


class ComplexityInvariantDistance(Feature):
    """
    A distance metric that accounts for signal complexity.

    Parameters
    ----------
    normalize : bool, optional
        Normalize the signal. Default is True.

    Methods
    -------
    compute(signal[, columns=None, windowed=False])
    """
    def __init__(self, normalize=True):
        super(ComplexityInvariantDistance, self).__init__(
            'ComplexityInvariantDistance', {'normalize': normalize}
        )
        self.normalize = normalize

    def _compute(self, x, fs):
        super(ComplexityInvariantDistance, self)._compute(x, fs)
        self._result = _cython.CID(x, self.normalize)


class RangeCountPercentage(Feature):
    """
    The percent of the signal that falls between specified values

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
    def __init__(self, range_min=-1.0, range_max=1.0):
        super(RangeCountPercentage, self).__init__(
            'RangeCountPercentage',
            {'range_min': range_min, 'range_max': range_max}
        )

        self.rmin = range_min
        self.rmax = range_max

    def _compute(self, x, fs):
        super(RangeCountPercentage, self)._compute(x, fs)

        self._result = _cython.RangeCount(x, self.rmin, self.rmax)


class RatioBeyondRSigma(Feature):
    """
    The percent of the signal outside :math:`r` standard deviations from the mean.

    Parameters
    ----------
    r : float, optional
        Number of standard deviations above or below the mean the range includes. Default is 2.0

    Methods
    -------
    compute(signal[, columns=None, windowed=False])
    """
    def __init__(self, r=2.0):
        super(RatioBeyondRSigma, self).__init__(
            'RatioBeyondRSigma',
            {'r': r}
        )

        self.r = r

    def _compute(self, x, fs):
        super(RatioBeyondRSigma, self)._compute(x, fs)

        self._result = _cython.RatioBeyondRSigma(x, self.r)
