"""
Different entropy measures

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from skdh.features.core import Feature
from skdh.features.lib import extensions

__all__ = ["SignalEntropy", "SampleEntropy", "PermutationEntropy"]


class SignalEntropy(Feature):
    r"""
    A Measure of the information contained in a signal. Also described as a measure of how
    surprising the outcome of a variable is.

    Notes
    -----
    The entropy is estimated using the histogram of the input signal. Bin limits for the
    histogram are defined per

    .. math::

        n_{bins} = ceil(\sqrt{N})
        \delta = \frac{x_{max} - x_{min}}{N - 1}
        bin_{min} = x_{min} - \frac{\delta}{2}
        bin_{max} = x_{max} + \frac{\delta}{2}

    where :math:`N` is the number of samples in the signal. Note that the data
    is standardized before computing (using mean and standard deviation).

    With the histogram, then the estimate of the entropy is computed per

    .. math::

        H_{est} = -\sum_{i=1}^kf(x_i)ln(f(x_i)) + ln(w) - bias
        w = \frac{bin_{max} - bin_{min}}{n_{bins}}
        bias = -\frac{n_{bins} - 1}{2N}

    Because of the standardization before the histogram computation, the entropy
    estimate is scaled again per

    .. math:: H_{est} = exp(H_{est}^2) - 2

    References
    ----------
    .. [1] Wallis, Kenneth. "A note on the calculation of entropy from histograms". 2006.
        https://warwick.ac.uk/fac/soc/economics/staff/academic/wallis/publications/entropy.pdf
    """

    __slots__ = ()

    def __init__(self):
        super(SignalEntropy, self).__init__()

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the signal entropy

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the signal entropy for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        sig_ent : numpy.ndarray
            Computed signal entropy.
        """
        x = super().compute(signal, axis=axis)

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
    .. [1] https://archive.physionet.org/physiotools/sampen/c/sampen.c
    """

    __slots__ = ("m", "r")

    def __init__(self, m=4, r=1.0):
        super(SampleEntropy, self).__init__(m=m, r=r)

        self.m = m
        self.r = r

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the sample entropy of a signal

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the sample entropy for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        samp_en : numpy.ndarray
            Computed sample entropy.
        """
        x = super().compute(signal, axis=axis)
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

    __slots__ = ("order", "delay", "normalize")

    def __init__(self, order=3, delay=1, normalize=False):
        super(PermutationEntropy, self).__init__(
            order=order, delay=delay, normalize=False
        )
        self.order = order
        self.delay = delay
        self.normalize = normalize

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the permutation entropy

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the signal entropy for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        perm_en : numpy.ndarray
            Computed permutation entropy.
        """
        x = super().compute(signal, axis=axis)
        return extensions.permutation_entropy(x, self.order, self.delay, self.normalize)
