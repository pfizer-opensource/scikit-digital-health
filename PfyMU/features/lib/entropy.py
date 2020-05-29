"""
Different entropy measures

Lukas Adamowicz
Pfizer DMTI 2020
"""
from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


class SignalEntropy(Feature):
    def __init__(self):
        """
        Compute the signal entropy of a given signal.

        Methods
        -------
        compute(signal[, columns=None])
        """