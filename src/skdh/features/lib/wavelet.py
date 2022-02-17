"""
Features using wavelet transforms

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import zeros, ceil, log2, sort, sum, diff, sign, maximum
import pywt

from skdh.features.core import Feature


__all__ = ["DetailPower", "DetailPowerRatio"]


class DetailPower(Feature):
    """
    The summed power in the detail levels that span the chosen frequency band.

    Parameters
    ----------
    wavelet : str
        Wavelet to use. Options are the discrete wavelets in `PyWavelets`.
        Default is 'coif4'.
    freq_band : array_like
        2-element array-like of the frequency band (Hz) to get the power in.
        Default is [1, 3].

    References
    ----------
    .. [1] Sekine, M. et al. "Classification of waist-acceleration signals in a
        continuous walking record." Medical Engineering & Physics. Vol. 22.
        Pp 285-291. 2000.
    """

    __slots__ = ("wave", "f_band")
    _wavelet_options = pywt.wavelist(kind="discrete")

    def __init__(self, wavelet="coif4", freq_band=None):
        super().__init__(wavelet=wavelet, freq_band=freq_band)

        self.wave = wavelet

        if freq_band is not None:
            self.f_band = sort(freq_band)
        else:
            self.f_band = [1.0, 3.0]

    def compute(self, signal, fs=1.0, *, axis=-1):
        """
        Compute the detail power

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the detail power for.
        fs : float, optional
            Sampling frequency in Hz. If not provided, default is 1.0Hz.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if
            `signal` is a pandas.DataFrame. Default is last (-1).

        Returns
        -------
        power : numpy.ndarray
            Computed detail power.
        """
        x = super().compute(signal, fs, axis=axis)

        # computation
        lvls = [
            int(ceil(log2(fs / self.f_band[0]))),  # maximum level needed
            int(ceil(log2(fs / self.f_band[1]))),  # minimum level to include in sum
        ]

        # TODO test effect of mode on result
        cA, *cD = pywt.wavedec(x, self.wave, mode="symmetric", level=lvls[0], axis=-1)

        # set non necessary levels to 0
        for i in range(lvls[0] - lvls[1] + 1, lvls[0]):
            cD[i][:] = 0.0

        # reconstruct and get negative->positive zero crossings
        xr = pywt.waverec((cA,) + tuple(cD), self.wave, mode="symmetric", axis=-1)

        N = sum(diff(sign(xr), axis=-1) > 0, axis=-1).astype(float)
        # ensure no 0 values to prevent divide by 0
        N = maximum(N, 1e-10)

        rshape = x.shape[:-1]

        result = zeros(rshape)
        for i in range(lvls[0] - lvls[1] + 1):
            result += sum(cD[i] ** 2, axis=-1)
        return result / N


class DetailPowerRatio(Feature):
    """
    The ratio of the power in the detail signals that span the specified
    frequency band. Uses the discrete wavelet transform to break down the
    signal into constituent components at different frequencies.

    Parameters
    ----------
    wavelet : str
        Wavelet to use. Options are the discrete wavelets in `PyWavelets`.
        Default is 'coif4'.
    freq_band : array_like
        2-element array-like of the frequency band (Hz) to get the power in.
        Default is [1, 10].

    Notes
    -----
    In the original paper [1]_, the result is multiplied by 100 to obtain a
    percentage. This final multiplication is not included in order to obtain
    results that have a scale that closer matches the typical 0-1 (or -1 to 1)
    scale for machine learning features. NOTE that this does not mean that
    the values will be in this range - since the scaling factor
    is the original acceleration and not the wavelet detail values.

    References
    ----------
    .. [1] Sekine, M. et al. "Classification of waist-acceleration signals in a
        continuous walking record." Medical Engineering & Physics. Vol. 22.
        Pp 285-291. 2000.
    """

    __slots__ = ("wave", "f_band")
    _wavelet_options = pywt.wavelist(kind="discrete")

    def __init__(self, wavelet="coif4", freq_band=None):
        super().__init__(wavelet=wavelet, freq_band=freq_band)

        self.wave = wavelet

        if freq_band is not None:
            self.f_band = sort(freq_band)
        else:
            self.f_band = [1.0, 10.0]

    def compute(self, signal, fs=1.0, *, axis=-1):
        """
        Compute the detail power ratio

        Parameters
        ----------
        signal : array-like
            Array-like containing values to compute the detail power ratio for.
        fs : float, optional
            Sampling frequency in Hz. If not provided, default is 1.0Hz.
        axis : int, optional
            Axis along which the signal entropy will be computed. Ignored if `signal` is a
            pandas.DataFrame. Default is last (-1).

        Returns
        -------
        power_ratio : numpy.ndarray
            Computed detail power ratio.
        """
        x = super().compute(signal, fs, axis=axis)

        # compute the required levels
        lvls = [
            int(ceil(log2(fs / self.f_band[0]))),  # maximum level needed
            int(ceil(log2(fs / self.f_band[1]))),  # minimum level to include in sum
        ]

        # TODO test effect of mode on result
        cA, *cD = pywt.wavedec(x, self.wave, mode="symmetric", level=lvls[0], axis=-1)

        result = zeros(x.shape[:-1])
        for i in range(lvls[0] - lvls[1] + 1):
            result += sum(cD[i] ** 2, axis=-1)

        return result / sum(x**2, axis=-1)
