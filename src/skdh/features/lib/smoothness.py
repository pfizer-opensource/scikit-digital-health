"""
Features dealing with the smoothness of a signal

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import log as nplog, abs

from skdh.features.core import Feature
from skdh.features.lib import extensions

__all__ = ["JerkMetric", "DimensionlessJerk", "SPARC"]


class JerkMetric(Feature):
    r"""
    The normalized sum of jerk.  Assumes the input signal is acceleration, and
    therefore the jerk is the first time derivative of the input signal.

    Notes
    -----
    Given an acceleration signal :math:`a`, the pre-normalized jerk metric
    :math:`\hat{J}` is computed using a 2-point difference of the acceleration,
    then squared and summed per

    .. math:: \hat{J} = \sum_{i=2}^N\left(\frac{a_{i} - a_{i-1}}{\Delta t}\right)^2

    where :math:`\Delta t` is the sampling period in seconds. The jerk metric
    :math:`J` is then normalized using constants and the maximum absolute
    acceleration value observed per

    .. math:: s = \frac{360max(|a|)^2}{\Delta t}
    .. math:: J = \frac{\hat{J}}{2s}
    """
    __slots__ = ()

    def __init__(self):
        super(JerkMetric, self).__init__()

    def compute(self, signal, fs=1.0, *, axis=-1):
        """
        Compute the jerk metric

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the jerk metric for.
        fs : float, optional
            Sampling frequency in Hz. If not provided, default is 1.0Hz
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if
            `signal` is a pandas.DataFrame. Default is last (-1).

        Returns
        -------
        jerk_metric : numpy.ndarray
            Computed jerk metric.
        """
        x = super().compute(signal, fs, axis=axis)
        return extensions.jerk_metric(x, fs)


class DimensionlessJerk(Feature):
    r"""
    The dimensionless normalized sum of jerk, or its log value. Will take
    velocity, acceleration, or jerk as the input signal, and compute the jerk
    accordingly.

    Parameters
    ----------
    log : bool, optional
        Take the log of the dimensionless jerk. Default is False.
    signal_type : {'acceleration', 'velocity', 'jerk'}, optional
        The type of the signal being provided. Default is 'acceleration'

    Notes
    -----
    For all three inputs (acceleration, velocity, and jerk) the squaring and
    summation of the computed jerk values is the same as :py:class:`JerkMetric`.
    The difference comes in the normalization to get a dimensionless value, and
    in the computation of the jerk.

    For the different inputs, the pre-normalized metric :math:`\hat{J}` is
    computed per

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

    Note that the sampling period ends up cancelling out for all versions of
    the metric. Finally, the dimensionless jerk metric is simply the negative
    pre-normalized value divided by the corresponding scaling factor. If the
    log dimensionless jerk is required, then the negative is taken after taking
    the natural logarithm

    .. math::

        DJ = \frac{-\hat{J}_{type}}{s_{type}} \\
        DJ_{log} = -ln\left(\frac{\hat{J}_{type}}{s_{type}}\right)
    """
    __slots__ = ("log", "i_type")
    _signal_type_options = ["velocity", "acceleration", "jerk"]

    def __init__(self, log=False, signal_type="acceleration"):
        super(DimensionlessJerk, self).__init__(log=log, signal_type=signal_type)

        self.log = log

        t_map = {"velocity": 1, "acceleration": 2, "jerk": 3}
        try:
            self.i_type = t_map[signal_type]
        except KeyError:
            raise ValueError(f"'signal_type' ({signal_type}) unrecognized.")

    def compute(self, signal, *, axis=-1, **kwargs):
        """
        compute(signal, *, axis=-1)

        Compute the dimensionless jerk metric

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the dimensionless jerk
            metric for.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if
            `signal` is a pandas.DataFrame. Default is last (-1).

        Returns
        -------
        dimless_jerk_metric : numpy.ndarray
            Computed [log] dimensionless jerk metric.
        """
        x = super().compute(signal, axis=axis)
        res = extensions.dimensionless_jerk_metric(x, self.i_type)

        if self.log:
            return -nplog(abs(res))
        else:
            return res


class SPARC(Feature):
    """
    A quantitative measure of the smoothness of a signal. SPARC stands for the
    SPectral ARC length.

    Parameters
    ----------
    padlevel : int
        Indicates the level of zero-padding to perform on the signal. This
        essentially multiplies the length of the signal by 2^padlevel. Default
        is 4.
    fc: float, optional
        The max. cut off frequency for calculating the spectral arc length
        metric. Default is 10.0 Hz.
    amplitude_threshold : float, optional
        The amplitude threshold to used for determining the cut off frequency
        up to which the spectral arc length is to be estimated. Default is 0.05


    References
    ----------
    .. [1] S. Balasubramanian, A. Melendez-Calderon, A. Roby-Brami, and
        E. Burdet, “On the analysis of movement smoothness,” J NeuroEngineering
        Rehabil, vol. 12, no. 1, p. 112, Dec. 2015, doi: 10.1186/s12984-015-0090-9.

    """

    __slots__ = ("padlevel", "fc", "amp_thresh")

    def __init__(self, padlevel=4, fc=10.0, amplitude_threshold=0.05):
        super(SPARC, self).__init__(
            padlevel=padlevel, fc=fc, amplitude_threshold=amplitude_threshold
        )

        self.padlevel = padlevel
        self.fc = fc
        self.amp_thresh = amplitude_threshold

    def compute(self, signal, fs=1.0, *, axis=-1):
        """
        compute(signal, fs, *, columns=None, windowed=False)

        Compute the SPARC

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the SPARC for.
        fs : float, optional
            Sampling frequency in Hz. If not provided, default is 1.0Hz
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if
            `signal` is a pandas.DataFrame. Default is last (-1).

        Returns
        -------
        sparc : numpy.ndarray
            Computed SPARC.
        """
        x = super().compute(signal, fs, axis=axis)
        return extensions.SPARC(x, fs, self.padlevel, self.fc, self.amp_thresh)
