"""
Different entropy measures

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import empty, arange, multiply, zeros, power, unique, log2
from math import factorial

from skimu.features.core import Feature
from skimu.features.lib import extensions

__all__ = ['SignalEntropy', 'SampleEntropy', 'PermutationEntropy']


class SignalEntropy(Feature):
    """
    A Measure of the information contained in a signal. Also described as a measure of how
    surprising the outcome of a variable is.
    """
    def __init__(self):
        super(SignalEntropy, self).__init__()

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the signal entropy

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the signal entropy for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        sig_ent : numpy.ndarray
            Computed signal entropy.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.signal_entropy(x)


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
        super(SampleEntropy, self).__init__(m=m, r=r)

        self.m = m
        self.r = r

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the sample entropy of a signal

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the sample entropy for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        samp_en : numpy.ndarray
            Computed sample entropy.
        """
        return super().compute(signal, fs=-1, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.sample_entropy(x, self.m, self.r)


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
            order=order,
            delay=delay,
            normalize=False
        )
        self.order = order
        self.delay = delay
        self.normalize = normalize

    def compute(self, signal, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the permutation entropy

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the signal entropy for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).
        col_axis : int, optional
            Axis along which column indexing will be done. Ignored if `signal` is a pandas.DataFrame
            or if `signal` is 2D.
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.

        Returns
        -------
        perm_en : numpy.ndarray
            Computed permutation entropy.
        """
        return super().compute(signal, fs=1., axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.permutation_entropy(x, self.order, self.delay, self.normalize)
