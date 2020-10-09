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
    """
    def __init__(self, normalize=True):
        super(ComplexityInvariantDistance, self).__init__(
            'ComplexityInvariantDistance', {'normalize': normalize}
        )
        self.normalize = normalize

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the complexity invariant distance

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
        cid : {numpy.ndarray, pandas.DataFrame}
            Computed complexity invariant distance, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

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
    """
    def __init__(self, range_min=-1.0, range_max=1.0):
        super(RangeCountPercentage, self).__init__(
            'RangeCountPercentage',
            {'range_min': range_min, 'range_max': range_max}
        )

        self.rmin = range_min
        self.rmax = range_max

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the range count percentage

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
        rcp : {numpy.ndarray, pandas.DataFrame}
            Computed range count percentage, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

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
    """
    def __init__(self, r=2.0):
        super(RatioBeyondRSigma, self).__init__(
            'RatioBeyondRSigma',
            {'r': r}
        )

        self.r = r

    def compute(self, *args, **kwargs):
        r"""
        compute(signal, *, columns=None, windowed=False)

        Compute the ratio beyond :math:`r\sigma`

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
        rbr : {numpy.ndarray, pandas.DataFrame}
            Computed ratio beyond r sigma, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(RatioBeyondRSigma, self)._compute(x, fs)

        self._result = _cython.RatioBeyondRSigma(x, self.r)
