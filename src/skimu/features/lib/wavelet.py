"""
Features using wavelet transforms

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import zeros, ceil, log2, sort, sum, diff, sign
import pywt

from skimu.features.core import Feature


__all__ = ['DetailPower', 'DetailPowerRatio']


class DetailPower(Feature):
    """
    The summed power in the detail levels that span the chosen frequency band.

    Parameters
    ----------
    wavelet : str
        Wavelet to use. Options are the discrete wavelets in `PyWavelets`. Default is 'coif4'
    freq_band : array_like
        2-element array-like of the frequency band (Hz) to get the power in. Default is [1, 3]

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])

    References
    ----------
    .. [1] Sekine, M. et al. "Classification of waist-acceleration signals in a continuous
    walking record." Medical Engineering & Physics. Vol. 22. Pp 285-291. 2000.
    """
    def __init__(self, wavelet='coif4', freq_band=None):
        super().__init__('DetailPower', {'wavelet': wavelet, 'freq_band': freq_band})

        self.wave = wavelet

        if freq_band is not None:
            self.f_band = sort(freq_band)
        else:
            self.f_band = [1.0, 3.0]

    def _compute(self, x, fs):
        super()._compute(x, fs)

        # computation
        lvls = [
            int(ceil(log2(fs / self.f_band[0]))),  # maximum level needed
            int(ceil(log2(fs / self.f_band[1])))  # minimum level to include in sum
        ]

        # TODO test effect of mode on result
        cA, *cD = pywt.wavedec(x, self.wave, mode='symmetric', level=lvls[0], axis=1)

        # set non necessary levels to 0
        for i in range(lvls[0] - lvls[1] + 1, lvls[0]):
            cD[i][:] = 0.

        # reconstruct and get negative->positive zero crossings
        xr = pywt.waverec((cA,) + tuple(cD), self.wave, mode='symmetric', axis=1)

        N = sum(diff(sign(xr), axis=1) > 0, axis=1).astype(float)
        # ensure no 0 values to prevent divide by 0
        N[N == 0] = 1e-4

        self._result = zeros((x.shape[0], x.shape[2]))
        for i in range(lvls[0] - lvls[1] + 1):
            self._result += sum(cD[i]**2, axis=1)
        self._result /= N


class DetailPowerRatio(Feature):
    """
    The ratio of the power in the detail signals that span the specified frequency band. Uses
    the discrete wavelet transform to break down the signal into constituent components at
    different frequencies.

    Parameters
    ----------
    wavelet : str
        Wavelet to use. Options are the discrete wavelets in `PyWavelets`. Default is 'coif4'
    freq_band : array_like
        2-element array-like of the frequency band (Hz) to get the power in. Default is [1, 10]

    Methods
    -------
    compute(signal, fs[, columns=None, windowed=False])

    Notes
    -----
    In the original paper [1]_, the result is multiplied by 100 to obtain a percentage. This
    final multiplication is not included in order to obtain results that have a scale that
    closer matches the typical 0-1 (or -1 to 1) scale for machine learning features.

    References
    ----------
    .. [1] Sekine, M. et al. "Classification of waist-acceleration signals in a continuous
    walking record." Medical Engineering & Physics. Vol. 22. Pp 285-291. 2000.
    """
    def __init__(self, wavelet='coif4', freq_band=None):
        super().__init__('DetailPowerRatio', {'wavelet': wavelet, 'freq_band': freq_band})

        self.wave = wavelet

        if freq_band is not None:
            self.f_band = sort(freq_band)
        else:
            self.f_band = [1.0, 10.0]

    def _compute(self, x, fs):
        super()._compute(x, fs)

        # compute the required levels
        lvls = [
            int(ceil(log2(fs / self.f_band[0]))),  # maximum level needed
            int(ceil(log2(fs / self.f_band[1])))  # minimum level to include in sum
        ]

        # TODO test effect of mode on result
        cA, *cD = pywt.wavedec(x, self.wave, mode='symmetric', level=lvls[0], axis=1)

        self._result = zeros((x.shape[0], x.shape[2]))
        for i in range(lvls[0] - lvls[1] + 1):
            self._result += sum(cD[i]**2, axis=1)

        self._result /= sum(x**2, axis=1)
