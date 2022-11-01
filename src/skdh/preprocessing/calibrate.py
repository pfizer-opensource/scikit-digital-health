"""
Inertial data/sensor calibration

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from warnings import warn

from numpy import (
    mean,
    diff,
    zeros,
    ones,
    abs,
    all as npall,
    around,
    Inf,
    vstack,
    minimum,
    concatenate,
)
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression

from skdh.base import BaseProcess
from skdh.utility import moving_mean, moving_sd


__all__ = ["CalibrateAccelerometer"]


class CalibrateAccelerometer(BaseProcess):
    """
    Calibrate pre-recording acceleration readings based on the deviation from 1G when motionless.
    Acceleration values can be modified in place. Calibration typically requires a minimum amount
    of data, which can be adjusted to more than the lower limit of 12 hours. If the minimum time
    specified is not enough, calibration will incrementally use more data until either the criteria
    are met, or all the data is used.

    Parameters
    ----------
    sphere_crit : float, optional
        Minimum acceleration value (in g) on both sides of 0g for each axis. Determines if the
        sphere is sufficiently populated to obtain a meaningful calibration result.
        Default is 0.3g.
    min_hours : int, optional
        Ideal minimum hours of data to use for the calibration. Any values not factors of 12 are
        rounded up to the nearest factor. Default is 72. If less than this amout of data is
        avialable (but still more than 12 hours), calibration will still be performed on all the
        data. If the calibration error is not under 0.01g after these hours, more data will be used
        in 12 hour increments.
    sd_criteria : float, optional
        The criteria for the rolling standard deviation to determine stillness, in g. This value
        will likely change between devices. Default is 0.013g, which was found for GeneActiv
        devices. If measuring the noise in a bench-top test, this threshold should be about
        `1.2 * noise`.
    max_iter : int, optional
        Maximum number of iterations to perform during calibration. Default is 1000. Generally
        should be left at this value.
    tol : float, optional
        Tolerance for stopping iteration. Default is 1e-10. Generally this should be left at this
        value.

    Notes
    -----
    This calibration relies on the assumption that a perfectly calibrated accelerometer's
    acceleration readings will lie on the unit sphere when motionless. Therefore, this calibration
    enforces that constraint on motionless data present in the recording to its best ability.

    References
    ----------
    .. [1] V. T. van Hees et al., “Autocalibration of accelerometer data for free-living physical
        activity assessment using local gravity and temperature: an evaluation on four continents,”
        Journal of Applied Physiology, vol. 117, no. 7, pp. 738–744, Aug. 2014,
        doi: 10.1152/japplphysiol.00421.2014.
    """

    def __init__(
        self, sphere_crit=0.3, min_hours=72, sd_criteria=0.013, max_iter=1000, tol=1e-10
    ):
        if min_hours % 12 != 0 or min_hours <= 0:
            min_hours = ((min_hours // 12) + 1) * 12

        max_iter = int(max_iter)

        super().__init__(
            sphere_crit=sphere_crit,
            min_hours=min_hours,
            sd_criteria=sd_criteria,
            max_iter=max_iter,
            tol=tol,
        )

        self.sphere_crit = sphere_crit
        self.min_hours = min_hours
        self.sd_crit = sd_criteria
        self.max_iter = max_iter
        self.tol = tol

    def predict(
        self, time=None, accel=None, *, fs=None, apply=True, temperature=None, **kwargs
    ):
        r"""
        Run the calibration on the accelerometer data.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by centrally mounted lumbar device, in
            units of 'g'.
        fs : float, optional
            Sampling frequency in Hz. If not provided, it is calculated from the
            timestamps.
        apply : bool, optional
            Apply the calibration to the acceleration. Default is True. Both cases return the
            scale, offset, and temperature scale in the return dictionary.
        temperature : numpy.ndarray
            (N, ) array of temperature measured by the sensor.

        Returns
        -------
        results : dictionary
            Returns input data, as well as the acceleration scale, acceleration offset, and
            temperature scale.

        Notes
        -----
        The scaling factors are applied as follows:

        .. math:: a_i(t) = (y_i(t) + d_i)s_i + (T(t) - \bar{T}_c)m_i

        where :math:`a_i` is the corrected acceleration in the *ith* axis, :math:`t` is a time
        point, :math:`y_i` is the measured accleration in the *ith* axis, :math:`d_i` is the offset
        for the *ith* axis, :math:`s_i` is the scale for the *ith* axis. If available, :math:`T` is
        the sensor measured temperature, :math:`\bar{T}_c` is the mean temperature from the
        calibration, and :math:`m_i` is the temperature scale for the *ith* axis.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            fs=fs,
            apply=apply,
            temperature=temperature,
            **kwargs,
        )

        # update before it might have to be returned early
        kwargs.update(
            {"fs": fs, self._time: time, self._acc: accel, self._temp: temperature}
        )

        # calculate fs if necessary
        fs = 1 / mean(diff(time)) if fs is None else fs
        # parameters
        n10 = int(10 * fs)  # samples in 10 seconds
        nh = int(self.min_hours * 3600 * fs)  # samples in min_hours
        n12h = int(12 * 3600 * fs)  # samples in 12 hours

        i_h = 0  # keep track of number of extra 12 hour blocks used
        if accel.shape[0] < nh:
            warn(
                f"Less than {self.min_hours} hours of data ({accel.shape[0] / (fs * 3600)} hours). "
                f"No Calibration performed",
                UserWarning,
            )
            kwargs.update({self._time: time, self._acc: accel, self._temp: temperature})
            return (kwargs, None) if self._in_pipeline else kwargs

        finished = False
        valid_calibration = True
        # use the Store object in order to save computation time
        store = Store(n10)
        while not finished:
            store.acc_rsd = accel[: nh + i_h * n12h]
            if temperature is not None:
                store.tmp_rm = temperature[: nh + i_h * n12h]
            else:
                store._tmp_rm = zeros(store._acc_rm.shape[0])

            (
                finished,
                offset,
                scale,
                temp_scale,
                temp_mean,
            ) = self._do_iterative_closest_point_fit(store)

            if not finished and (nh + i_h * n12h) >= accel.shape[0]:
                finished = True
                valid_calibration = False
                warn(
                    f"Recalibration not done with {self.min_hours + i_h * 12} hours due to "
                    f"insufficient non-movement data available"
                )
            i_h += 1

        if apply and valid_calibration:
            if temperature is None:
                accel = (accel + offset) * scale
            else:
                accel = (accel + offset) * scale + (temperature - temp_mean)[
                    :, None
                ] * temp_scale

        # add the results to the returned values
        kwargs.update(
            {
                self._acc: accel,
                "offset": offset,
                "scale": scale,
                "temperature scale": temp_scale,
            }
        )

        return (kwargs, None) if self._in_pipeline else kwargs

    def _do_iterative_closest_point_fit(self, store):
        """
        Perform the minimization using an iterative closest point fitting procedure

        Parameters
        ----------
        store : Store
            Stored rolling mean/sd values needed for the computation. Used in order to cut down
            on computation time.

        Returns
        -------
        finished : bool
            If the optimization finished successfully.
        offset : numpy.ndarray
            (3, ) array of offsets
        scale : numpy.ndarray
            (3, ) array of scaling factors
        tmp_scale : numpy.ndarray
            (1, 3) array of scaling factors for the temperature
        tmp_mean : float
            Mean temperature from the calibration
        """
        # initialize offset, scale, and tmp_scale
        offset = zeros(3)
        scale = ones(3)
        tmp_scale = zeros((1, 3))

        # get parts with no motion. <2 is to prevent clipped signals from being labeled
        no_motion = npall(store.acc_rsd < self.sd_crit, axis=1) & npall(
            abs(store.acc_rm) < 2, axis=1
        )
        # nans are automatically excluded

        # trim to no motion only
        acc_rsd = store.acc_rsd[no_motion]
        acc_rm = store.acc_rm[no_motion]
        tmp_rm = store.tmp_rm[no_motion]

        # make sure enough points
        if acc_rsd.shape[0] < 2:
            return False, offset, scale, tmp_scale, 0.0

        # starting error
        cal_error_start = around(mean(abs(norm(acc_rm, axis=1) - 1)), decimals=5)

        # check if the sphere is well populated
        tel = (
            (acc_rm.min(axis=0) < -self.sphere_crit)
            & (acc_rm.max(axis=0) > self.sphere_crit)
        ).sum()

        if tel != 3:
            return False, offset, scale, tmp_scale, 0.0

        # calibration
        tmp_mean = mean(tmp_rm)
        tmp_rm = (tmp_rm - tmp_mean).reshape((-1, 1))

        weights = ones(acc_rm.shape[0]) * 100
        res = [Inf]
        LR = LinearRegression()

        for niter in range(self.max_iter):
            curr = (acc_rm + offset) * scale + tmp_rm @ tmp_scale

            closest_point = curr / norm(curr, axis=1, keepdims=True)
            offsetch = zeros(3)
            scalech = ones(3)
            toffch = zeros((1, 3))

            for k in range(3):
                # there was some code dropping NANs from closest point, but these should
                # be taken care of in the original mask. Division by zero should also
                # not be happening during motionless data, where 1 value should always be close
                # to 1
                x_ = vstack(
                    (curr[:, k], tmp_rm[:, 0])
                ).T  # don't need the ones in Python
                LR.fit(x_, closest_point[:, k], sample_weight=weights)

                offsetch[k] = LR.intercept_
                scalech[k] = LR.coef_[0]
                toffch[0, k] = LR.coef_[1]
                curr[:, k] = x_ @ LR.coef_

            offset = offset + offsetch / (scale * scalech)
            tmp_scale = tmp_scale * scalech + toffch
            scale = scale * scalech

            res.append(
                3 * mean(weights[:, None] * (curr - closest_point) ** 2 / weights.sum())
            )
            weights = minimum(1 / norm(curr - closest_point, axis=1), 100)

            # tolerance condition during iteration
            if abs(res[niter] - res[niter - 1]) < self.tol:
                break

        # end of iteration (break or end) calibration error assessment
        acc_rm = (acc_rm + offset) * scale + tmp_rm * tmp_scale
        cal_error_end = around(mean(abs(norm(acc_rm, axis=1) - 1)), decimals=5)

        # assess if calibration error has been significantly improved
        if (cal_error_end < cal_error_start) and (cal_error_end < 0.01):
            return True, offset, scale, tmp_scale, tmp_mean
        else:
            return False, offset, scale, tmp_scale, tmp_mean


class Store:
    """
    Class for storing moving SD and mean values for update
    """

    __slots__ = ("_acc_rsd", "_acc_rm", "_tmp_rm", "_n", "_nt", "wlen")

    def __init__(self, wlen):
        self._acc_rsd = None
        self._acc_rm = None
        self._tmp_rm = None
        self._n = 0
        self._nt = 0

        self.wlen = wlen

    @property
    def acc_rsd(self):
        return self._acc_rsd

    @acc_rsd.setter
    def acc_rsd(self, value):
        if self._acc_rsd is None:
            self._acc_rsd, self._acc_rm = moving_sd(
                value, self.wlen, self.wlen, axis=0, return_previous=True
            )
            self._n = int((value.shape[0] // self.wlen) * self.wlen)
        else:
            _rsd, _rm = moving_sd(
                value[self._n :], self.wlen, self.wlen, axis=0, return_previous=True
            )
            self._acc_rsd = concatenate((self._acc_rsd, _rsd), axis=0)
            self._acc_rm = concatenate((self._acc_rm, _rm), axis=0)
            self._n += int((value[self._n :].shape[0] // self.wlen) * self.wlen)

    @property
    def acc_rm(self):
        return self._acc_rm

    @property
    def tmp_rm(self):
        return self._tmp_rm

    @tmp_rm.setter
    def tmp_rm(self, value):
        if self._tmp_rm is None:
            self._tmp_rm = moving_mean(value, self.wlen, self.wlen)
            self._nt = int((value.shape[0] // self.wlen) * self.wlen)
        else:
            _rm = moving_mean(value[self._nt :], self.wlen, self.wlen)
            self._tmp_rm = concatenate((self._tmp_rm, _rm), axis=0)
            self._nt += int((value[self._nt :].shape[0] // self.wlen) * self.wlen)
