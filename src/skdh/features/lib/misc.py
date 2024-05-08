"""
Misc features

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from skdh.features.core import Feature
from skdh.features.lib import extensions

__all__ = ["ComplexityInvariantDistance", "RangeCountPercentage", "RatioBeyondRSigma"]


class ComplexityInvariantDistance(Feature):
    """
    A distance metric that accounts for signal complexity.

    Parameters
    ----------
    normalize : bool, optional
        Normalize the signal. Default is True.
    """

    __slots__ = ("normalize",)

    def __init__(self, normalize=True):
        super(ComplexityInvariantDistance, self).__init__(normalize=normalize)
        self.normalize = normalize

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the complexity invariant distance

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the complexity invariant
            distance for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if
            `signal` is a pandas.DataFrame. Default is last (-1).

        Returns
        -------
        cid : numpy.ndarray
            Computed complexity invariant distance.
        """
        x = super().compute(signal, axis=axis)
        return extensions.complexity_invariant_distance(x, self.normalize)


class RangeCountPercentage(Feature):
    """
    The percent of the signal that falls between specified values

    Parameters
    ----------
    range_min : {int, float}, optional
        Minimum value of the range. Default value is -1.0
    range_max : {int, float}, optional
        Maximum value of the range. Default value is 1.0
    """

    __slots__ = ("rmin", "rmax")

    def __init__(self, range_min=-1.0, range_max=1.0):
        super(RangeCountPercentage, self).__init__(
            range_min=range_min, range_max=range_max
        )

        self.rmin = range_min
        self.rmax = range_max

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the range count percentage

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the range count percentage for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        rcp : numpy.ndarray
            Computed range count percentage.
        """
        x = super().compute(signal, fs=1.0, axis=axis)
        return extensions.range_count(x, self.rmin, self.rmax)


class RatioBeyondRSigma(Feature):
    """
    The percent of the signal outside :math:`r` standard deviations from the mean.

    Parameters
    ----------
    r : float, optional
        Number of standard deviations above or below the mean the range includes. Default is 2.0
    """

    __slots__ = ("r",)

    def __init__(self, r=2.0):
        super(RatioBeyondRSigma, self).__init__(r=r)

        self.r = r

    def compute(self, signal, *, axis=-1, **kwargs):
        r"""
        compute(signal, *, axis=-1)

        Compute the ratio beyond :math:`r\sigma`

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the ratio beyond :math:`r\sigma` for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        rbr : numpy.ndarray
            Computed ratio beyond r sigma.
        """
        x = super().compute(signal, fs=1.0, axis=axis)
        return extensions.ratio_beyond_r_sigma(x, self.r)
