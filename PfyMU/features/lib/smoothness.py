"""
Features dealing with the smoothness of a signal

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import log, abs

from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


class JerkMetric(Feature):
    def __init__(self, normalize=True):
        """
        Compute a metric of the jerk of the signal.  Assumes the input signal is acceleration

        Methods
        -------
        compute(signal, fs[, columns=None])
        """
        super(JerkMetric, self).__init__('JerkMetric', {})
        self.normalize = normalize

    def _compute(self, x, fs):
        super(JerkMetric, self)._compute(x, fs)
        self._result = _cython.JerkMetric(x, fs)


class DimensionlessJerk(Feature):
    def __init__(self, log=False, signal_type='acceleration'):
        """
        Compute metric of the jerk of the signal, but dimensionless, or its log value. Can input velocity, acceleration,
        or jerk as the signal.

        Parameters
        ----------
        log : bool, optional
            Take the log of the dimensionless jerk. Default is False.
        signal_type : {'acceleration', 'velocity', 'jerk'}, optional
            The type of the signal being provided. Default is 'acceleration'

        Methods
        -------
        compute(signal[, columns=None])
        """
        super(DimensionlessJerk, self).__init__('DimensionlessJerk', {'log': log, 'signal_type': signal_type})

        self.log = log

        t_map = {'velocity': 1, 'acceleration': 2, 'jerk': 3}
        try:
            self.i_type = t_map[signal_type]
        except KeyError:
            raise ValueError(f"'signal_type' ({signal_type}) unrecognized.")

    def _compute(self, x, fs):
        super(DimensionlessJerk, self)._compute(x, fs)

        if self.log:
            self._result = -log(abs(_cython.DimensionlessJerk(x, self.i_type)))
        else:
            self._result = _cython.DimensionlessJerk(x, self.i_type)


class SPARC(Feature):
    def __init__(self, padlevel=4, fc=10.0, amplitude_threshold=0.05):
        """
        SPARC (SPectral ARC length) is a quantitative measure of the smoothness of a signal.

        Parameters
        ----------
        padlevel : int
            Indicates the level of zero-padding to perform on the signal. This essentially multiplies the length of the
            signal by 2^padlevel. Default is 4
        fc: float, optional
            The max. cut off frequency for calculating the spectral arc length metric. Default is 10.0 Hz.
        amplitude_threshold : float, optional
            The amplitude threshold to used for determining the cut off frequency up to which the spectral arc length is
            to be estimated. Default is 0.05

        Methods
        -------
        compute(signal, fs[, columns=None])


        References
        ----------
        S. Balasubramanian, A. Melendez-Calderon, A. Roby-Brami, E. Burdet. "On the analysis of movement smoothness."
            Journal of NeuroEngineering and Rehabilitation. 2015.
        """
        super(SPARC, self).__init__('SPARC', {'padlevel': padlevel, 'fc': fc,
                                              'amplitude_threshold': amplitude_threshold})
        self.padlevel = padlevel
        self.fc = fc
        self.amp_thresh = amplitude_threshold

    def _compute(self, x, fs):
        super(SPARC, self)._compute(x, fs)

        self._result = _cython.SPARC(x, fs, self.padlevel, self.fc, self.amp_thresh)
