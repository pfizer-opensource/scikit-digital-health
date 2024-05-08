"""
Function for getting strides from detected gait events

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from numpy import (
    max,
    min,
    mean,
    arccos,
    sum,
    array,
    sin,
    cos,
    full,
    nan,
    arctan2,
    unwrap,
    pi,
    sign,
    diff,
    abs,
    zeros,
    cross,
)
from numpy.linalg import norm

from skdh.utility.internal import rle


def get_turns(gait, accel, gyro, fs, n_strides):
    """
    Get the location of turns, to indicate if steps occur during a turn.

    Parameters
    ----------
    gait : dictionary
        Dictionary of gait values needed for computation or the results.
    accel : numpy.ndarray
        Acceleration in units of 'g', for the current gait bout.
    gyro : numpy.ndarray
        Angular velocity in units of 'rad/s', for the current gait bout.
    fs : float
        Sampling frequency, in Hz.
    n_strides : int
        Number of strides in the current gait bout.

    Notes
    -----
    Values indicate turns as follows:

    - -1: Turns not detected (lacking angular velocity data)
    - 0: No turn found
    - 1: Turn overlaps with either Initial or Final contact
    - 2: Turn overlaps with both Initial and Final contact

    References
    ----------
    .. [1] M. H. Pham et al., “Algorithm for Turning Detection and Analysis
        Validated under Home-Like Conditions in Patients with Parkinson’s Disease
        and Older Adults using a 6 Degree-of-Freedom Inertial Measurement Unit at
        the Lower Back,” Front. Neurol., vol. 8, Apr. 2017,
        doi: 10.3389/fneur.2017.00135.

    """
    # first check if we can detect turns
    if gyro is None or n_strides < 1:
        gait["Turn"].extend([-1] * n_strides)
        return

    # get the first available still period to start the yaw tracking
    n = int(0.05 * fs)  # number of samples to use for still period

    min_slice = None
    for i in range(int(2 * fs)):
        tmp = norm(accel[i : i + n], axis=1)
        acc_range = max(tmp) - min(tmp)
        if acc_range < (0.2 / 9.81):  # range defined by the Pham paper
            min_slice = accel[i : i + n]
            break

    if min_slice is None:
        min_slice = accel[:n]

    # compute the mean value over that time frame
    acc_init = mean(min_slice, axis=0)
    # compute the initial angle between this vector and global frame
    phi = arccos(sum(acc_init * array([0, 0, 1])) / norm(acc_init))

    # create the rotation matrix/rotations from sensor frame to global frame
    gsZ = array([sin(phi), cos(phi), 0.0])
    gsX = array([1.0, 0.0, 0.0])

    gsY = cross(gsZ, gsX)
    gsY /= norm(gsY)
    gsX = cross(gsY, gsZ)
    gsX /= norm(gsX)

    gsR = array([gsX, gsY, gsZ])

    # iterate over the gait bout
    alpha = full(gyro.shape[0], nan)  # allocate the yaw angle around vertical axis
    alpha[0] = arctan2(gsR[2, 0], gsR[1, 0])

    for i in range(1, gyro.shape[0]):
        theta = norm(gyro[i]) / fs
        c = cos(theta)
        s = sin(theta)
        t = 1 - c

        wx = gyro[i, 0]
        wy = gyro[i, 1]
        wz = gyro[i, 2]

        update_R = array(
            [
                [t * wx**2 + c, t * wx * wy + s * wz, t * wx * wz - s * wy],
                [t * wx * wy - s * wz, t * wy**2 + c, t * wy * wz + s * wx],
                [t * wx * wz + s * wy, t * wy * wz - s * wx, t * wz**2 + c],
            ]
        )

        gsR = update_R @ gsR
        alpha[i] = arctan2(gsR[2, 0], gsR[1, 0])

    # unwrap the angle so there are no discontinuities
    alpha = unwrap(alpha, period=pi)

    # get the sign of the difference as initial turn indication
    turns = sign(diff(alpha))

    # get the angles of the turns
    lengths, starts, values = rle(turns == 1)
    turn_angles = abs(alpha[starts + lengths] - alpha[starts])

    # find hesitations in turns
    mask = (lengths / fs) < 0.5  # less than half a second
    mask[1:-1] &= turn_angles[:-2] >= (pi / 180 * 10)  # adjacent turns > 10 degrees
    mask[1:-1] &= turn_angles[2:] >= (pi / 180 * 10)

    # one adjacent turn greater than 45 degrees
    mask[1:-1] &= (turn_angles[:-2] > pi / 4) | (turn_angles[2:] >= pi / 4)

    # magnitude of hesitation less than 10% of turn angle
    mask[1:-1] = turn_angles[1:-1] < (0.1 * (turn_angles[:-2] + turn_angles[2:]))

    # set hesitation turns to match surrounding
    for l, s in zip(lengths[mask], starts[mask]):
        turns[s : s + l] = turns[s - 1]

    # enforce the time limit (0.1 - 10s) and angle limit (90 deg)
    lengths, starts, values = rle(turns == 1)
    mask = abs(alpha[starts + lengths] - alpha[starts]) < (pi / 2)  # exclusion mask
    mask |= ((lengths / fs) < 0.1) & ((lengths / fs) > 10)
    for l, s in zip(lengths[mask], starts[mask]):
        turns[s : s + l] = 0
    # final list of turns
    lengths, starts, values = rle(turns != 0)

    # mask for strides in turn
    in_turn = zeros(n_strides, dtype="int")
    for d, s in zip(lengths[values == 1], starts[values == 1]):
        in_turn += (gait["IC"][-n_strides:] > s) & (gait["IC"][-n_strides:] < (s + d))
        in_turn += (gait["FC"][-n_strides:] > s) & (gait["FC"][-n_strides:] < (s + d))

    gait["Turn"].extend(in_turn)
