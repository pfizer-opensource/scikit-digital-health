"""
Compute the activity index from acceleration data

Yiorgos Christakis, Lukas Adamowicz
Pfizer DMTI 2019-2021
"""
from scipy.signal import butter, sosfiltfilt
from numpy import sqrt, mean, var, ascontiguousarray

from skimu.utility import get_windowed_view


def calculate_activity_index(fs, accel, hp_cut=0.25):
    # high pass filter
    sos = butter(3, hp_cut * 2 / fs, btype="high", output="sos")
    accel_hf = ascontiguousarray(sosfiltfilt(sos, accel, axis=0))

    # non-overlapping 60s windows
    acc_w = get_windowed_view(accel_hf, int(60 * fs), int(60 * fs))

    # compute activity index
    act_ind = sqrt(mean(var(acc_w, axis=2), axis=1))

    return act_ind
