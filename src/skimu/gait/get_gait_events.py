"""
Function for getting gait events from an accelerometer signal

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import fft, argmax, std, abs, round
from scipy.signal import detrend, butter, sosfiltfilt, find_peaks
from scipy.integrate import cumtrapz
from pywt import cwt


def get_gait_events(vert_accel, dt, va_sign, o_scale, filter_order, filter_cutoff,
                    use_optimal_scale):
    """
    Get the bouts of gait from the acceleration during a gait bout

    Parameters
    ----------
    vert_accel : numpy.ndarray
        (N, ) array of vertical acceleration during the gait bout
    dt : float
        Sampling period for the acceleration
    va_sign : int
        Sign of the vertical acceleration
    o_scale : int
        Original scale for the CWT
    filter_order : int
        Low-pass filter order
    filter_cutoff : float
        Low-pass filter cutoff in Hz
    use_optimal_scale : bool
        Use the optimal scale based on step frequency

    Returns
    -------
    init_contact : numpy.ndarray
        Indices of initial contacts
    final_contact : numpy.ndarray
        Indices of final contacts
    vert_accel : numpy.ndarray
        Filtered vertical acceleration
    vert_velocity : numpy.ndarray
        Vertical velocity
    vert_position : numpy.ndarray
        Vertical position
    """
    vert_accel = detrend(vert_accel)  # detrend data just in case

    # low-pass filter
    sos = butter(filter_order, 2 * filter_cutoff * dt, btype='low', output='sos')
    filt_vert_accel = sosfiltfilt(sos, vert_accel)

    # first integrate the vertical acceleration to get vertical velocity
    vert_velocity = cumtrapz(filt_vert_accel, dx=dt, initial=0)
    # integrate again for future use, after detrending
    vert_position = cumtrapz(detrend(vert_velocity), dx=dt, initial=0)

    # if using the optimal scale relationship, get the optimal scale
    if use_optimal_scale:
        coef_scale_original, _ = cwt(vert_velocity, o_scale, 'gaus1')
        F = abs(fft.rfft(coef_scale_original[0]))
        # compute an estimate of the step frequency
        step_freq = argmax(F) / vert_velocity.size / dt

        ic_opt_freq = 0.69 * step_freq + 0.34
        fc_opt_freq = 3.6 * step_freq - 4.5

        scale1 = round(0.4 / (2 * ic_opt_freq * dt)) - 1
        scale2 = round(0.4 / (2 * fc_opt_freq * dt)) - 1
    else:
        scale1 = scale2 = o_scale

    coef1, _ = cwt(vert_velocity, [scale1, scale2], 'gaus1')
    """
    Find the local minima in the signal. This should technically always require using
    the negative signal in "find_peaks", however the way PyWavelets computes the
    CWT results in the opposite signal that we want.
    Therefore, if the sign of the acceleration was negative, we need to use the
    positve coefficient signal, and opposite for positive acceleration reading.
    """
    init_contact, *_ = find_peaks(-va_sign * coef1[0], height=0.5 * std(coef1[0]))

    coef2, _ = cwt(coef1[1], scale2, 'gaus1')

    """
    Peaks are the final contact points
    Same issue as above
    """
    final_contact, *_ = find_peaks(-va_sign * coef2[0], height=0.5 * std(coef2[0]))

    return init_contact, final_contact, filt_vert_accel, vert_velocity, vert_position
