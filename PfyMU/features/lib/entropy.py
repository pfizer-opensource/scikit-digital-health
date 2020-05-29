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
        super(SignalEntropy, self).__init__('SampleEntropy', {})

    def _compute(self, x, fs):
        super(SignalEntropy, self)._compute(x, fs)

        self._result = _cython.SignalEntropy(x)


class SampleEntropy(Feature):
    def __init__(self, m=4, r=1.0):
        """
        Compute the sample entropy of a given signal, which is the negative log that if the distance between two sets
        of `m` points is less than `r`, then the distance between two sets of `m+1` points is also less than `r`

        Parameters
        ----------
        m : int, optional
            Set length for comparison. Default is 4
        r : float, optional
            Maximum distance between sets. Default is 1.0

        Methods
        -------
        compute(signal[, columns=None])

        Notes
        -----
        The distance metric used is the Chebyshev distance, which is defined as the maximum absolute value of the
        sample-by-sample difference between two sets of the same length

        References
        ----------
        https://archive.physionet.org/physiotools/sampen/c/sampen.c
        """
        super(SampleEntropy, self).__init__('SampleEntropy', {'m': m, 'r': r})

        self.m = m
        self.r = r

    def _compute(self, x, fs):
        super(SampleEntropy, self)._compute(x, fs)

        # TODO check computation
        res = _cython.SampleEntropy(x, self.m, self.r)

        self._result = res[:, -1, :]

