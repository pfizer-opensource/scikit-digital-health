"""
Different entropy measures

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import empty, arange, multiply, zeros, power, unique, log2
from math import factorial

from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


__all__ = ['SignalEntropy', 'SampleEntropy', 'PermutationEntropy']


class SignalEntropy(Feature):
    def __init__(self):
        """
        Compute the signal entropy of a given signal.

        Methods
        -------
        compute(signal[, columns=None, windowed=False])
        """
        super(SignalEntropy, self).__init__('SignalEntropy', {})

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
        compute(signal[, columns=None, windowed=False])

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


class PermutationEntropy(Feature):
    def __init__(self, order=3, delay=1, normalize=False):
        """
        Calculate permutation entropy of signals. Permutation entropy is a meausure of the signal complexity based on
        how the temporal signal behaves according to a series of ordinal patterns.

        Parameters
        ----------
        order : int, optional
            Order (length of sub-signals) to use in the computation. Default is 3
        delay : int, optional
            Time-delay to use in computing the sub-signals. Default is 1 sample.
        normalize : bool, optional
            Normalize the output between 0 and 1. Default is False.

        Methods
        -------
        compute(signal, fs[, columns=None, windowed=False])
        """
        super(PermutationEntropy, self).__init__("PermutationEntropy", {'order': order, 'delay': delay,
                                                                        'normalize': normalize})
        self.order = order
        self.delay = delay
        self.normalize = normalize

    def _compute(self, x, fs):
        super(PermutationEntropy, self)._compute(x, fs)

        pe = zeros((x.shape[0], x.shape[2]))
        hashmult = power(self.order, arange(self.order))

        for wind in range(x.shape[0]):
            for ax in range(x.shape[2]):
                # Embed x and sort the order of permutations
                sorted_idx = PermutationEntropy._embed(x[wind, :, ax], self.order, self.delay).argsort(kind='quicksort')

                # Associate unique integer to each permutations
                hashval = (multiply(sorted_idx, hashmult)).sum(1)

                # Return the counts
                _, c = unique(hashval, return_counts=True)

                # Use true_divide for Python 2 compatibility
                p = c / c.sum()
                pe[wind, ax] = -multiply(p, log2(p)).sum()
        if self.normalize:
            pe = pe / log2(factorial(self.order))

        self._result = pe

    @staticmethod
    def _embed(x_1d, order, delay):
        """
        Time-delay embedding.

        Parameters
        ----------
        x_1d : 1d-array, shape (n_times)
            Time series
        order : int
            Embedding dimension (order)
        delay : int
            Delay.
        Returns
        -------
        embedded : ndarray, shape (n_times - (order - 1) * delay, order)
            Embedded time-series.
        """
        N = x_1d.size
        Y = empty((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x_1d[i * delay:i * delay + Y.shape[1]]
        return Y.T

