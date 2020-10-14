"""
Features dealing with the smoothness of a signal

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import log, abs

from skimu.features.core import Feature
from skimu.features.lib import _cython

__all__ = ['JerkMetric', 'DimensionlessJerk', 'SPARC']


class JerkMetric(Feature):
    r"""
    The normalized sum of jerk.  Assumes the input signal is acceleration, and therefore the jerk
    is the first time derivative of the input signal.

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])

    Notes
    -----
    Given an acceleration signal :math:`a`, the pre-normalized jerk metric :math:`\hat{J}` is
    computed using a 2-point difference of the acceleration, then squared and summed per

    .. math:: \hat{J} = \sum_{i=2}^N\left(\frac{a_{i} - a_{i-1}}{\Delta t}\right)^2

    where :math:`\Delta t` is the sampling period in seconds. The jerk metric :math:`J` is then
    normalized using constants and the maximum absolute acceleration value observed per

    .. math:: s = \frac{360max(|a|)^2}{\Delta t}
    .. math:: J = \frac{\hat{J}}{2s}
    """
    def __init__(self, normalize=True):
        super(JerkMetric, self).__init__('JerkMetric', {})
        self.normalize = normalize

    def compute(self, *args, **kwargs):
        """
        compute(signal, fs, *, columns=None, windowed=False)

        Compute the jerk metric

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
        jerk_metric : {numpy.ndarray, pandas.DataFrame}
            Computed jerk metric, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(JerkMetric, self)._compute(x, fs)
        self._result = _cython.JerkMetric(x, fs)


class DimensionlessJerk(Feature):
    r"""
    The dimensionless normalized sum of jerk, or its log value. Will take velocity, acceleration,
    or jerk as the input signal, and compute the jerk accordingly.

    Parameters
    ----------
    log : bool, optional
        Take the log of the dimensionless jerk. Default is False.
    signal_type : {'acceleration', 'velocity', 'jerk'}, optional
        The type of the signal being provided. Default is 'acceleration'

    Methods
    -------
    compute(signal[, columns=None, windowed=False])

    Notes
    -----
    For all three inputs (acceleration, velocity, and jerk) the squaring and summation of the
    computed jerk values is the same as :py:class:`JerkMetric`. The difference comes in the
    normalization to get a dimensionless value, and in the computation of the jerk.

    For the different inputs, the pre-normalized metric :math:`\hat{J}` is computed per

    .. math::

        \hat{J}_{vel} = \sum_{i=2}^{N-1}\left(\frac{v_{i+1} - 2v_{i}
            + v_{i-1}}{\Delta t^2}\right)^2 \\
        \hat{J}_{acc} = \sum_{i=2}^N\left(\frac{a_{i} - a_{i-1}}{\Delta t}\right)^2 \\
        \hat{J}_{jerk} = \sum_{i=1}^Nj_i^2

    The scaling factor also changes depending on which input is provided, per

    .. math::

        s_{vel} = \frac{max(|v|)^2}{N^3\Delta t^4} \\
        s_{acc} = \frac{max(|a|)^2}{N \Delta t^2} \\
        s_{jerk} = Nmax(|j|)^2

    Note that the sampling period ends up cancelling out for all versions of the metric. Finally,
    the dimensionless jerk metric is simply the negative pre-normalized value divided by
    the corresponding scaling factor. If the log dimensionless jerk is required, then the negative
    is taken after taking the natural logarithm

    .. math::

        DJ = \frac{-\hat{J}_{type}}{s_{type}} \\
        DJ_{log} = -ln\left(\frac{\hat{J}_{type}}{s_{type}}\right)
    """
    def __init__(self, log=False, signal_type='acceleration'):
        super(DimensionlessJerk, self).__init__(
            'DimensionlessJerk', {'log': log, 'signal_type': signal_type}
        )

        self.log = log

        t_map = {'velocity': 1, 'acceleration': 2, 'jerk': 3}
        try:
            self.i_type = t_map[signal_type]
        except KeyError:
            raise ValueError(f"'signal_type' ({signal_type}) unrecognized.")

    def compute(self, *args, **kwargs):
        """
        compute(signal, *, columns=None, windowed=False)

        Compute the dimensionless jerk metric

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
        dimless_jerk_metric : {numpy.ndarray, pandas.DataFrame}
            Computed [log] dimensionless jerk metric, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(DimensionlessJerk, self)._compute(x, fs)

        if self.log:
            self._result = -log(abs(_cython.DimensionlessJerk(x, self.i_type)))
        else:
            self._result = _cython.DimensionlessJerk(x, self.i_type)


class SPARC(Feature):
    """
    A quantitative measure of the smoothness of a signal. SPARC stands for the SPectral
    ARC length

    Parameters
    ----------
    padlevel : int
        Indicates the level of zero-padding to perform on the signal. This essentially
        multiplies the length of the signal by 2^padlevel. Default is 4
    fc: float, optional
        The max. cut off frequency for calculating the spectral arc length metric. Default is
        10.0 Hz.
    amplitude_threshold : float, optional
        The amplitude threshold to used for determining the cut off frequency up to which the
        spectral arc length is to be estimated. Default is 0.05

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])


    References
    ----------
    .. [1] S. Balasubramanian, A. Melendez-Calderon, A. Roby-Brami, and E. Burdet, “On the
        analysis of movement smoothness,” J NeuroEngineering Rehabil, vol. 12, no. 1, p. 112,
        Dec. 2015, doi: 10.1186/s12984-015-0090-9.

    """
    def __init__(self, padlevel=4, fc=10.0, amplitude_threshold=0.05):
        super(SPARC, self).__init__('SPARC', {'padlevel': padlevel, 'fc': fc,
                                              'amplitude_threshold': amplitude_threshold})
        self.padlevel = padlevel
        self.fc = fc
        self.amp_thresh = amplitude_threshold

    def compute(self, *args, **kwargs):
        """
        compute(signal, fs, *, columns=None, windowed=False)

        Compute the SPARC

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
        sparc : {numpy.ndarray, pandas.DataFrame}
            Computed SPARC, returned as the same type as the input signal
        """
        return super().compute(*args, **kwargs)

    def _compute(self, x, fs):
        super(SPARC, self)._compute(x, fs)

        self._result = _cython.SPARC(x, fs, self.padlevel, self.fc, self.amp_thresh)