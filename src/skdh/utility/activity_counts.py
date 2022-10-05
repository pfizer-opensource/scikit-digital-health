"""
Computation of activity counts.

Lukas Adamowicz
Copyright (c) 2022. Pfizer Inc. All rights reserved.
"""
from numpy import array, around, repeat, abs, minimum, floor, float_
from scipy.signal import lfilter_zi, lfilter

from skdh.utility.internal import apply_downsample
from skdh.utility import moving_mean


input_coef = array(
    [
        [
            -0.009341062898525,
            -0.025470289659360,
            -0.004235264826105,
            0.044152415456420,
            0.036493718347760,
            -0.011893961934740,
            -0.022917390623150,
            -0.006788163862310,
            0.000000000000000,
        ]
    ],
    dtype=float_,
)

output_coef = array(
    [
        [
            1.00000000000000000000,
            -3.63367395910957000000,
            5.03689812757486000000,
            -3.09612247819666000000,
            0.50620507633883000000,
            0.32421701566682000000,
            -0.15685485875559000000,
            0.01949130205890000000,
            0.00000000000000000000,
        ]
    ],
    dtype=float_
)


def get_activity_counts(fs, time, accel, epoch_seconds=60):
    """
    Compute the activity counts from acceleration.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    time : numpy.ndarray
        Shape (N,) array of epoch timestamps (in seconds) for each sample.
    accel : numpy.ndarray
        Nx3 array of measured acceleration values, in units of g.
    epoch_seconds : int, optional
        Number of seconds in an epoch (time unit for counts). Default is 60 seconds.

    Returns
    -------
    counts : numpy.ndarray
        Array of activity counts
    """
    # 1. resample to 30hz
    acc_ds = apply_downsample(
        30.0,
        time,
        data=(accel,),
        aa_filter=True,
        fs=fs,
    )

    acc_ds = around(acc_ds, decimals=3)

    # filter the data
    # NOTE: this is the actigraph implementation - they specifically use
    # a filter with a phase shift (ie not filtfilt), and TF representation
    # instead of ZPK or SOS
    zi = lfilter_zi(input_coef, output_coef).reshape((-1, 1))

    acc_bpf, _ = lfilter(
        input_coef,
        output_coef,
        acc_ds,
        zi=repeat(zi, acc_ds.shape[1], axis=-1) * acc_ds[0]
    )

    # scale the data
    acc_bpf *= (3.0 / 4096.) / (2.6 / 256.) * 237.5

    # trim the data
    acc_trim = abs(acc_bpf)
    acc_trim[acc_trim < 4] = 0
    acc_trim = floor(minimum(acc_trim, 128))

    # "downsample" to 10hz by taking moving mean
    acc_10hz = moving_mean(acc_trim, 3, 3, trim=True, axis=0)

    # get the counts
    block_size = epoch_seconds * 10  # 1 minute
    # this time is a moving sum
    epoch_counts = moving_mean(acc_10hz, block_size, block_size, trim=True, axis=0)
    epoch_counts *= block_size  # remove the "mean" part to get back to sum

    # floor the counts
    epoch_counts = floor(epoch_counts)

    return epoch_counts
