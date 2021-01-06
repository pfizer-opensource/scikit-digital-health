"""
Frequency based features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from skimu.features.core import Feature
from skimu.features.lib import extensions

__all__ = ['DominantFrequency', 'DominantFrequencyValue', 'PowerSpectralSum', 'SpectralFlatness',
           'SpectralEntropy']


class DominantFrequency(Feature):
    r"""
    The primary frequency in the signal. Computed using the FFT and finding the maximum value of
    the power spectral density in the specified range of frequencies.

    Parameters
    ----------
    padlevel : int, optional
        Padding (factors of 2) to use in the FFT computation. Default is 2.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    The `padlevel` parameter effects the number of points to be used in the FFT computation by
    factors of 2. The computation of number of points is per

    .. math:: nfft = 2^{ceil(log_2(N)) + padlevel}

    So `padlevel=2` would mean that for a signal with length 150, the number of points used
    in the FFT would go from 256 to 1024.
    """
    def __init__(self, padlevel=2, low_cutoff=0.0, high_cutoff=5.0):
        super(DominantFrequency, self).__init__(
            padlevel=padlevel,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.padlevel = padlevel
        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def compute(self, signal, fs, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the dominant frequency

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the dominant frequency for.
        fs : float
            Sampling frequency in Hz.
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
        dom_freq : numpy.ndarray
            Computed dominant frequency.
        """
        return super().compute(signal, fs, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.dominant_frequency(x, fs, self.padlevel, self.low_cut, self.high_cut)


class DominantFrequencyValue(Feature):
    r"""
    The power spectral density maximum value. Taken inside the range of frequencies specified.

    Parameters
    ----------
    padlevel : int, optional
        Padding (factors of 2) to use in the FFT computation. Default is 2.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    The `padlevel` parameter effects the number of points to be used in the FFT computation by
    factors of 2. The computation of number of points is per

    .. math:: nfft = 2^{ceil(log_2(N)) + padlevel}

    So `padlevel=2` would mean that for a signal with length 150, the number of points used
    in the FFT would go from 256 to 1024.
    """
    def __init__(self, padlevel=2, low_cutoff=0.0, high_cutoff=5.0):
        super(DominantFrequencyValue, self).__init__(
            padlevel=padlevel,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.padlevel = padlevel
        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def compute(self, signal, fs, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the dominant frequency value

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the dominant frequency value for.
        fs : float
            Sampling frequency in Hz.
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
        dom_freq_val : numpy.ndarray
            Computed dominant frequency value.
        """
        return super().compute(signal, fs, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.dominant_frequency_value(x, fs, self.padlevel, self.low_cut, self.high_cut)


class PowerSpectralSum(Feature):
    r"""
    Sum of power spectral density values. The sum of power spectral density values in a
    1.0Hz wide band around the primary (dominant) frequency (:math:`f_{dom}\pm 0.5`)

    Parameters
    ----------
    padlevel : int, optional
        Padding (factors of 2) to use in the FFT computation. Default is 2.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    The `padlevel` parameter effects the number of points to be used in the FFT computation by
    factors of 2. The computation of number of points is per

    .. math:: nfft = 2^{ceil(log_2(N)) + padlevel}

    So `padlevel=2` would mean that for a signal with length 150, the number of points used
    in the FFT would go from 256 to 1024.
    """
    def __init__(self, padlevel=2, low_cutoff=0.0, high_cutoff=5.0):
        super(PowerSpectralSum, self).__init__(
            padlevel=padlevel,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.padlevel = padlevel
        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def compute(self, signal, fs, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the power spectral sum

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the power spectral sum for.
        fs : float
            Sampling frequency in Hz.
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
        pss : numpy.ndarray
            Computed power spectral sum.
        """
        return super().compute(signal, fs, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.power_spectral_sum(x, fs, self.padlevel, self.low_cut, self.high_cut)


class SpectralFlatness(Feature):
    r"""
    A measure of the "tonality" or resonant structure of a signal. Provides a quantification of
    how tone-like a signal is, as opposed to being noise-like. For this case, tonality is defined
    in a sense as the amount of peaks in the power spectrum, opposed to a flat signal representing
    white noise.

    Parameters
    ----------
    padlevel : int, optional
        Padding (factors of 2) to use in the FFT computation. Default is 2.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    The `padlevel` parameter effects the number of points to be used in the FFT computation by
    factors of 2. The computation of number of points is per

    .. math:: nfft = 2^{ceil(log_2(N)) + padlevel}

    So `padlevel=2` would mean that for a signal with length 150, the number of points used
    in the FFT would go from 256 to 1024.
    """
    def __init__(self, padlevel=2, low_cutoff=0.0, high_cutoff=5.0):
        super(SpectralFlatness, self).__init__(
            padlevel=padlevel,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.padlevel = padlevel
        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def compute(self, signal, fs, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the spectral flatness

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the spectral flatness for.
        fs : float
            Sampling frequency in Hz.
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
        spec_flat : numpy.ndarray
            Computed spectral flatness.
        """
        return super().compute(signal, fs, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.spectral_flatness(x, fs, self.padlevel, self.low_cut, self.high_cut)


class SpectralEntropy(Feature):
    r"""
    A measure of the information contained in the power spectral density estimate. Similar
    to :py:class:`SignalEntropy` but for the power spectral density.

    Parameters
    ----------
    padlevel : int, optional
        Padding (factors of 2) to use in the FFT computation. Default is 2.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    The `padlevel` parameter effects the number of points to be used in the FFT computation by
    factors of 2. The computation of number of points is per

    .. math:: nfft = 2^{ceil(log_2(N)) + padlevel}

    So `padlevel=2` would mean that for a signal with length 150, the number of points used
    in the FFT would go from 256 to 1024.
    """
    def __init__(self, padlevel=2, low_cutoff=0.0, high_cutoff=5.0):
        super(SpectralEntropy, self).__init__(
            padlevel=padlevel,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.padlevel = padlevel
        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def compute(self, signal, fs, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the spectral entropy

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the spectral entropy for.
        fs : float
            Sampling frequency in Hz.
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
        spec_ent : numpy.ndarray
            Computed spectral entropy.
        """
        return super().compute(signal, fs, axis=axis, col_axis=col_axis, columns=columns)

    def _compute(self, x, fs):
        return extensions.spectral_entropy(x, fs, self.padlevel, self.low_cut, self.high_cut)
