"""
Frequency based features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import array

from PfyMU.features.core import Feature
from PfyMU.features.lib import _cython


class DominantFrequency(Feature):
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        """
        Compute the dominant frequency in the range specified

        Parameters
        ----------
        low_cutoff : float, optional
            Low value of the frequency range to look in. Default is 0.0 Hz
        high_cutoff : float, optional
            High value of the frequency range to look in. Default is 5.0 Hz

        Methods
        -------
        compute(signal, fs[, columns=None])
        """
        super(DominantFrequency, self).__init__('DominantFrequency', {'low_cutoff': low_cutoff,
                                                                      'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(DominantFrequency, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_dominant_freq(x, fs, self.low_cut, self.high_cut))


class DominantFrequencyValue(Feature):
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        """
        Compute the maximum power spectral density estimate in the specified range of frequencies.

        Parameters
        ----------
        low_cutoff : float, optional
            Low value of the frequency range to look in. Default is 0.0 Hz
        high_cutoff : float, optional
            High value of the frequency range to look in. Default is 5.0 Hz

        Methods
        -------
        compute(signal, fs[, columns=None])
        """
        super(DominantFrequencyValue, self).__init__('DominantFrequencyValue', {'low_cutoff': low_cutoff,
                                                                                'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(DominantFrequencyValue, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_dominant_freq_value(x, fs, self.low_cut, self.high_cut))


class PowerSpectralSum(Feature):
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        """
        Compute sum of the power spectral density estimate in a 1.0Hz band around the dominant frequency

        Parameters
        ----------
        low_cutoff : float, optional
            Low value of the frequency range to look in. Default is 0.0 Hz
        high_cutoff : float, optional
            High value of the frequency range to look in. Default is 5.0 Hz

        Methods
        -------
        compute(signal, fs[, columns=None])
        """
        super(PowerSpectralSum, self).__init__('PowerSpectralSum', {'low_cutoff': low_cutoff,
                                                                    'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(PowerSpectralSum, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_power(x, fs, self.low_cut, self.high_cut))


class SpectralFlatness(Feature):
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        """
        Compute the spectral flatness in a range of frequencies. The spectral flatness is a measure of the
        "tonality" or resonant structure of a signal (as opposed to just noise).

        Parameters
        ----------
        low_cutoff : float, optional
            Low value of the frequency range to look in. Default is 0.0 Hz
        high_cutoff : float, optional
            High value of the frequency range to look in. Default is 5.0 Hz

        Methods
        -------
        compute(signal, fs[, columns=None])
        """
        super(SpectralFlatness, self).__init__('SpectralFlatness', {'low_cutoff': low_cutoff,
                                                                    'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(SpectralFlatness, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_spectral_flatness(x, fs, self.low_cut, self.high_cut))


class SpectralEntropy(Feature):
    def __init__(self, low_cutoff=0.0, high_cutoff=5.0):
        """
        Compute the spectral entropy in a specified frequency range. Spectral entropy is a measure of the information
        contained in the power spectral density estimate.

        Parameters
        ----------
        low_cutoff : float, optional
            Low value of the frequency range to look in. Default is 0.0 Hz
        high_cutoff : float, optional
            High value of the frequency range to look in. Default is 5.0 Hz

        Methods
        -------
        compute(signal, fs[, columns=None])
        """
        super(SpectralEntropy, self).__init__('SpectralEntropy', {'low_cutoff': low_cutoff,
                                                                  'high_cutoff': high_cutoff})

        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def _compute(self, x, fs):
        super(SpectralEntropy, self)._compute(x, fs)

        ff = _cython.FrequencyFeatures()

        self._result = array(ff.get_spectral_entropy(x, fs, self.low_cut, self.high_cut))
