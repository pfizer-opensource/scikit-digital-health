"""
Different entropy measures

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import empty, arange, multiply, zeros, power, unique, log2
from math import factorial

from skimu.features.core import Feature
from skimu.features.lib import _cython

__all__ = ['SignalEntropy', 'SampleEntropy', 'PermutationEntropy']


class SignalEntropy(Feature):
    """
    A Measure of the information contained in a signal. Also described as a measure of how
    surprising the outcome of a variable is.
    """
    def __init__(self):
        super(SignalEntropy, self).__init__('SignalEntropy', {})

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the signal entropy

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
        sig_en : {numpy.ndarray, pandas.DataFrame}
            Computed signal entropy, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(SignalEntropy, self)._compute(x, fs)

        self._result = _cython.SignalEntropy(x)


class SampleEntropy(Feature):
    r"""
    A measure of the complexity of a time-series signal. Sample entropy is a modification of
    approximate entropy, but has the benefit of being data-length independent and having
    an easier implementation. Smaller values indicate more self-similarity in the dataset,
    and/or less noise.

    Parameters
    ----------
    m : int, optional
        Set length for comparison (aka embedding dimension). Default is 4
    r : float, optional
        Maximum distance between sets. Default is 1.0

    Notes
    -----
    Sample entropy first computes the probability that if two sets of length :math:`m`
    simultaneous data points have distance :math:`<r`, then two sets of length :math:`m+`
    simultaneous data points also have distance :math:`<r`, and then takes the negative
    natural logarithm of this probability.

    .. math:: E_{sample} = -ln\frac{A}{B}

    where :math:`A=`number of :math:`m+1` vector pairs with distance :math:`<r`
    and :math:`B=`number of :math:`m` vector pairs with distance :math:`<r`

    The distance metric used is the Chebyshev distance, which is defined as the maximum
    absolute value of the sample-by-sample difference between two sets of the same length

    References
    ----------
    https://archive.physionet.org/physiotools/sampen/c/sampen.c
    """
    def __init__(self, m=4, r=1.0):
        super(SampleEntropy, self).__init__('SampleEntropy', {'m': m, 'r': r})

        self.m = m
        self.r = r

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the detail power ratio

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
        samp_en : {numpy.ndarray, pandas.DataFrame}
            Computed sample entropy, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(SampleEntropy, self)._compute(x, fs)

        # TODO check computation
        res = _cython.SampleEntropy(x, self.m, self.r)

        self._result = res[:, -1, :]


class PermutationEntropy(Feature):
    """
    A meausure of the signal complexity. Based on how the temporal signal behaves according to
    a series of ordinal patterns.

    Parameters
    ----------
    order : int, optional
        Order (length of sub-signals) to use in the computation. Default is 3
    delay : int, optional
        Time-delay to use in computing the sub-signals. Default is 1 sample.
    normalize : bool, optional
        Normalize the output between 0 and 1. Default is False.
    """
    def __init__(self, order=3, delay=1, normalize=False):
        super(PermutationEntropy, self).__init__(
            "PermutationEntropy", {'order': order, 'delay': delay, 'normalize': normalize}
        )
        self.order = order
        self.delay = delay
        self.normalize = normalize

    def compute(self, *args, **kwargs):
        """
        compute(signal, fs, *, columns=None, windowed=False)

        Compute the permutation entropy

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
        perm_en : {numpy.ndarray, pandas.DataFrame}
            Computed permutation entropy, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(PermutationEntropy, self)._compute(x, fs)

        pe = zeros((x.shape[0], x.shape[2]))
        hashmult = power(self.order, arange(self.order))

        for wind in range(x.shape[0]):
            for ax in range(x.shape[2]):
                # Embed x and sort the order of permutations
                sorted_idx = PermutationEntropy._embed(
                    x[wind, :, ax], self.order, self.delay
                ).argsort(kind='quicksort')

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
