"""
Utility functions dealing with IMU orientation

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from warnings import warn

from numpy import argmax, abs, mean, cos, arcsin, sign, zeros_like


__all__ = ["correct_accelerometer_orientation"]


def correct_accelerometer_orientation(accel, v_axis=None, ap_axis=None):
    r"""
    Applies the correction for acceleration from [1]_ to better align acceleration with the human
    body anatomical axes. This correction requires that the original device measuring accleration
    is somewhat closely aligned with the anatomical axes already, due to required assumptions.
    Quality of the correction will degrade the farther from aligned the input acceleration is.

    Parameters
    ----------
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g".
    v_axis : {None, int}, optional
        Vertical axis for `accel`. If not provided (default of None), this will be guessed as the
        axis with the largest mean value.
    ap_axis : {None, int}, optional
        Anterior-posterior axis for `accel`. If not provided (default of None), the ML and AP axes
        will not be picked. This will have a slight effect on the correction.

    Returns
    -------
    co_accel : numpy.ndarray
        (N, 3) array of acceleration with best alignment to the human anatomical axes

    Notes
    -----
    If `v_axis` is not provided (`None`), it is guessed as the largest mean valued axis (absolute
    value). While this should work for most cases, it will fail if there is significant
    acceleration in the non-vertical axes. As such, if there are very large accelerations present,
    this value should be provided.

    If `ap_axis` is not provided, it is guessed as the axis with the most similar autocovariance
    to the vertical axes.

    The correction algorithm from [1]_ starts by using simple trigonometric identities to correct
    the measured acceleration per

    .. math::

        a_A = a_a\cos{\theta_a} - sign(a_v)a_v\sin{\theta_a}
        a_V' = sign(a_v)a_a\sin{\theta_a} + a_v\cos{\theta_a}
        a_M = a_m\cos{\theta_m} - sign(a_v)a_V'\sin{\theta_m}
        a_V = sign(a_v)a_m\sin{\theta_m} + a_V'\cos{\theta_m}

    where $a_i$ is the measured $i$ direction acceleration, $a_I$ is the corrected $I$ direction
    acceleration ($i/I=[a/A, m/M, v/V]$, $a$ is anterior-posterior, $m$ is medial-lateral, and
    $v$ is vertical), $a_V'$ is a provisional estimate of the corrected vertical acceleration.
    $\theta_{a/m}$ are the angles between the measured AP and ML axes and the horizontal plane.

    Through some manipulation, [1]_ arrives at the simplification that best estimates for these
    angles per

    .. math::

        \sin{\theta_a} = \bar{a}_a
        \sin{\theta_m} = \bar{a}_m

    This is the part of the step that requires acceleration to be in "g", as well as mostly
    already aligned. If significantly out of alignment, then this small-angle relationship
    with sine starts to fall apart, and the correction will not be as appropriate.

    References
    ----------
    .. [1] R. Moe-Nilssen, “A new method for evaluating motor control in gait under real-life
        environmental conditions. Part 1: The instrument,” Clinical Biomechanics, vol. 13, no.
        4–5, pp. 320–327, Jun. 1998, doi: 10.1016/S0268-0033(98)00089-8.
    """
    if v_axis is None:
        v_axis = argmax(abs(mean(accel, axis=0)))
    else:
        if not (0 <= v_axis < 3):
            raise ValueError("v_axis must be in {0, 1, 2}")
    if ap_axis is None:
        ap_axis, ml_axis = [i for i in range(3) if i != v_axis]
    else:
        if not (0 <= ap_axis < 3):
            raise ValueError("ap_axis must be in {0, 1, 2}")
        ml_axis = [i for i in range(3) if i not in [v_axis, ap_axis]][0]

    s_theta_a = mean(accel[:, ap_axis])
    s_theta_m = mean(accel[:, ml_axis])
    # make sure the theta values are in range
    if s_theta_a < -1 or s_theta_a > 1 or s_theta_m < -1 or s_theta_m > 1:
        warn("Accel. correction angles outside possible range [-1, 1]. Not correcting.")
        return accel

    c_theta_a = cos(arcsin(s_theta_a))
    c_theta_m = cos(arcsin(s_theta_m))

    v_sign = sign(mean(accel[:, v_axis]))

    co_accel = zeros_like(accel)
    # correct ap axis acceleration
    co_accel[:, ap_axis] = (
        accel[:, ap_axis] * c_theta_a - v_sign * accel[:, v_axis] * s_theta_a
    )
    # provisional correction for vertical axis
    co_accel[:, v_axis] = (
        v_sign * accel[:, ap_axis] * s_theta_a + accel[:, v_axis] * c_theta_a
    )
    # correct ml axis acceleration
    co_accel[:, ml_axis] = (
        accel[:, ml_axis] * c_theta_m - v_sign * co_accel[:, v_axis] * s_theta_m
    )
    # final correction for vertical axis
    co_accel[:, v_axis] = (
        v_sign * accel[:, ml_axis] * s_theta_m + co_accel[:, v_axis] * c_theta_m
    )

    return co_accel
