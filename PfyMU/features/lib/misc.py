"""
Misc features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import max, min, quantile, mean, std, arange

from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


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
        compute(signal[, columns=None])
        """
        super(ComplexityInvariantDistance, self).__init__('ComplexityInvariantDistance', {'Normalize': normalize})
        self.normalize = normalize

    def _compute(self, x, fs):
        self._result = _cython.CID(x, self.normalize)
