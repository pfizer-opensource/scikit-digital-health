"""
Frequency based features

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import array

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
    nfft : int, optional
        Number of points in the FFT to use. Default is 1024.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    While the `nfft` parameter allows specification of arbitrary points to be used in the FFT,
    fastest results will be for powers of 2. Additionally, there is a minimum `nfft` defined as

    .. math:: NFFT_{min} = 2^{log_2(N)}

    where `N` is the number of points in the computation axis. If `nfft` is set below this, it
    will be automatically bumped up, without warning.
    """
    def __init__(self, nfft=1024, low_cutoff=0.0, high_cutoff=5.0):
        super(DominantFrequency, self).__init__(
            nfft=nfft,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.nfft = nfft
        self.low_cut = low_cutoff
        self.high_cut = high_cutoff

    def compute(self, signal, fs, *, axis=-1, col_axis=-1, columns=None):
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
        return extensions.dominant_frequency(x, fs, self.nfft, self.low_cut, self.high_cut)


class DominantFrequencyValue(Feature):
    r"""
    The power spectral density maximum value. Taken inside the range of frequencies specified.

    Parameters
    ----------
    nfft : int, optional
        Number of points in the FFT to use. Default is 1024.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    While the `nfft` parameter allows specification of arbitrary points to be used in the FFT,
    fastest results will be for powers of 2. Additionally, there is a minimum `nfft` defined as

    .. math:: NFFT_{min} = 2^{log_2(N)}

    where `N` is the number of points in the computation axis. If `nfft` is set below this, it
    will be automatically bumped up, without warning.
    """
    def __init__(self, nfft=1024, low_cutoff=0.0, high_cutoff=5.0):
        super(DominantFrequencyValue, self).__init__(
            nfft=nfft,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.nfft = nfft
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
        return extensions.dominant_frequency_value(x, fs, self.nfft, self.low_cut, self.high_cut)


class PowerSpectralSum(Feature):
    r"""
    Sum of power spectral density values. The sum of power spectral density values in a
    1.0Hz wide band around the primary (dominant) frequency (:math:`f_{dom}\pm 0.5`)

    Parameters
    ----------
    nfft : int, optional
        Number of points in the FFT to use. Default is 1024.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    While the `nfft` parameter allows specification of arbitrary points to be used in the FFT,
    fastest results will be for powers of 2. Additionally, there is a minimum `nfft` defined as

    .. math:: NFFT_{min} = 2^{log_2(N)}

    where `N` is the number of points in the computation axis. If `nfft` is set below this, it
    will be automatically bumped up, without warning.
    """
    def __init__(self, nfft=1024, low_cutoff=0.0, high_cutoff=5.0):
        super(PowerSpectralSum, self).__init__(
            nfft=nfft,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.nfft = nfft
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
        return extensions.power_spectral_sum(x, fs, self.nfft, self.low_cut, self.high_cut)


class SpectralFlatness(Feature):
    r"""
    A measure of the "tonality" or resonant structure of a signal. Provides a quantification of
    how tone-like a signal is, as opposed to being noise-like. For this case, tonality is defined
    in a sense as the amount of peaks in the power spectrum, opposed to a flat signal representing
    white noise.

    Parameters
    ----------
    nfft : int, optional
        Number of points in the FFT to use. Default is 1024.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    While the `nfft` parameter allows specification of arbitrary points to be used in the FFT,
    fastest results will be for powers of 2. Additionally, there is a minimum `nfft` defined as

    .. math:: NFFT_{min} = 2^{log_2(N)}

    where `N` is the number of points in the computation axis. If `nfft` is set below this, it
    will be automatically bumped up, without warning.
    """
    def __init__(self, nfft=1024, low_cutoff=0.0, high_cutoff=5.0):
        super(SpectralFlatness, self).__init__(
            nfft=nfft,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.nfft = nfft
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
        return extensions.spectral_flatness(x, fs, self.nfft, self.low_cut, self.high_cut)


class SpectralEntropy(Feature):
    r"""
    A measure of the information contained in the power spectral density estimate. Similar
    to :py:class:`SignalEntropy` but for the power spectral density.

    Parameters
    ----------
    nfft : int, optional
        Number of points in the FFT to use. Default is 1024.
    low_cutoff : float, optional
        Low value of the frequency range to look in. Default is 0.0 Hz
    high_cutoff : float, optional
        High value of the frequency range to look in. Default is 5.0 Hz

    Notes
    -----
    While the `nfft` parameter allows specification of arbitrary points to be used in the FFT,
    fastest results will be for powers of 2. Additionally, there is a minimum `nfft` defined as

    .. math:: nfft_{min} = 2^{log_2(N)}

    where `N` is the number of points in the computation axis. If `nfft` is set below this, it
    will be automatically bumped up, without warning.
    """
    def __init__(self, nfft=1024, low_cutoff=0.0, high_cutoff=5.0):
        super(SpectralEntropy, self).__init__(
            nfft=nfft,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )

        self.nfft = nfft
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
        return extensions.spectral_entropy(x, fs, self.nfft, self.low_cut, self.high_cut)
