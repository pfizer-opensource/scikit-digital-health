"""
Frequency based features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import array

from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython

__all__ = ['DominantFrequency', 'DominantFrequencyValue', 'PowerSpectralSum', 'SpectralFlatness',
           'SpectralEntropy']


class DominantFrequency(Feature):
    """
    The primary frequency in the signal. Computed using the FFT and finding the maximum value of
    the power spectral density in the specified range of frequencies.

    Parameters
    ----------
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])
    """
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        super(DominantFrequency, self).__init__('DominantFrequency', {'low_cutoff': low_cutoff,
                                                                      'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(DominantFrequency, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_dominant_freq(x, fs, self.low_cut, self.high_cut))


class DominantFrequencyValue(Feature):
    """
    The power spectral density maximum value. Taken inside the range of frequencies specified.

    Parameters
    ----------
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])
    """
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        super(DominantFrequencyValue, self).__init__(
            'DominantFrequencyValue', {'low_cutoff': low_cutoff, 'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(DominantFrequencyValue, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_dominant_freq_value(x, fs, self.low_cut, self.high_cut))


class PowerSpectralSum(Feature):
    r"""
    Sum of power spectral density values. The sum of power spectral density values in a
    1.0Hz wide band around the primary (dominant) frequency (:math:`f_{dom}\pm 0.5`)

    Parameters
    ----------
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])
    """
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        super(PowerSpectralSum, self).__init__('PowerSpectralSum', {'low_cutoff': low_cutoff,
                                                                    'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(PowerSpectralSum, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_power(x, fs, self.low_cut, self.high_cut))


class SpectralFlatness(Feature):
    """
    A measure of the "tonality" or resonant structure of a signal. Provides a quantification of
    how tone-like a signal is, as opposed to being noise-like. For this case, tonality is defined
    in a sense as the amount of peaks in the power spectrum, opposed to a flat signal representing
    white noise.

    Parameters
    ----------
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])
    """
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        super(SpectralFlatness, self).__init__('SpectralFlatness', {'low_cutoff': low_cutoff,
                                                                    'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(SpectralFlatness, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_spectral_flatness(x, fs, self.low_cut, self.high_cut))


class SpectralEntropy(Feature):
    """
    A measure of the information contained in the power spectral density estimate. Similar
    to :py:class:`SignalEntropy` but for the power spectral density.

    Parameters
    ----------
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])
    """
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        super(SpectralEntropy, self).__init__('SpectralEntropy', {'low_cutoff': low_cutoff,
                                                                  'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(SpectralEntropy, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_spectral_entropy(x, fs, self.low_cut, self.high_cut))
