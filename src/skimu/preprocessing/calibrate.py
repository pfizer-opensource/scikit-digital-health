"""
Inertial data/sensor calibration

Lukas Adamowicz
Pfizer DMTI 2021
"""
from warnings import warn

from numpy import mean, diff, zeros, ones, abs, all as npall, around, Inf, sum, vstack, minimum
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression

from skimu.base import _BaseProcess
from skimu.utility import rolling_mean, rolling_sd


__all__ = ["AccelerometerCalibrate"]


class AccelerometerCalibrate(_BaseProcess):
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
    def __init__(self, sphere_crit=0.3, min_hours=72, sd_criteria=0.013, max_iter=1000, tol=1e-10):
        if min_hours % 12 != 0 or min_hours <= 0:
            min_hours = ((min_hours // 12) + 1) * 12

        max_iter = int(max_iter)

        super().__init__(
            sphere_crit=sphere_crit,
            min_hours=min_hours,
            sd_criteria=sd_criteria,
            max_iter=max_iter,
            tol=tol
        )

        self.sphere_crit = sphere_crit
        self.min_hours = min_hours
        self.sd_crit = sd_criteria
        self.max_iter = max_iter
        self.tol = tol

    def predict(self, time=None, accel=None, *, apply=True, temperature=None, **kwargs):
        super().predict(
            time=time, accel=accel, apply=apply, temperature=temperature, **kwargs
        )

        # parameters
        fs = 1 / mean(diff(time[1000:5000]))  # dont start at zero due to timestamp weirdness
        n10 = int(10 * fs)  # samples in 10 seconds
        nh = int(self.min_hours * 3600 * fs)  # samples in min_hours
        n12h = int(12 * 3600 * fs)  # samples in 12 hours

        i_h = 0  # keep track of number of extra 12 hour blocks used
        if accel.shape[0] < nh:
            warn(f"Less than {self.min_hours} of data ({accel.shape[0] / (fs * 3600)}. "
                 f"No Calibration performed", UserWarning)
            kwargs.update({self._time: time, self._acc: accel, self._temp: temperature})
            return kwargs

        finished = False
        while not finished:
            finished, offset, scale, temp_scale, temp_mean = self.do_iterative_closest_point_fit(
                accel, temperature, nh + i_h * n12h, n10)

            if not finished and (nh + i_h * n12h) > accel.shape[0]:
                finished = True
                warn(f"Recalibration not done with {self.min_hours + i_h * 12} hours due to "
                     f"insufficient non-movement data available")
            i_h += 1

        if apply:
            if temperature is None:
                accel = (accel + offset) * scale
            else:
                accel = (accel + offset) * scale + (temperature - temp_mean) * temp_scale

        # add the results to the returned values
        kwargs.update({"offset": offset, "scale": scale, "temperature scale": temp_scale})

        kwargs.update({self._time: time, self._acc: accel, self._temp: temperature})
        return kwargs

    # def predict1(self, time=None, accel=None, *, apply=True, temp=None, **kwargs):
    #     super().predict(
    #         time=time, accel=accel, apply=apply, temp=temp, **kwargs
    #     )
    #
    #     # various parameters
    #     fs = 1 / mean(diff(time[:500]))
    #     n10 = int(10 / fs)
    #     nh = int(self.min_hours * 3600 / fs)  # elements in min_hours
    #     n12h = int(12 * 3600 / fs)  # elements in 12 hours
    #
    #     i_h = 0  # number of extra 12 hour blocks used
    #     if accel.shape[0] < nh:
    #         warn(f"Less than {self.min_hours} of data ({accel.shape[0] / fs / 3600}). "
    #              f"No Calibration performed")
    #         kwargs.update({self._time: time, self._acc: accel, self._temp: temp})
    #         return kwargs
    #
    #     while ((nh + n12h * i_h) - accel.shape[0]) < n12h or i_h == 0:
    #         accel_rsd, accel_rm = rolling_sd(
    #             accel[0:nh + n12h * i_h], n10, n10, axis=0, return_previous=True)
    #         mag_rm = rolling_mean(norm(accel[0:nh + n12h * i_h], axis=1), n10, n10)
    #
    #         if temp is not None:
    #             temp_rm = rolling_mean(temp[0:nh + n12h * i_h], n10, n10)
    #         else:
    #             temp_rm = zeros(accel_rsd.shape[0])
    #
    #         # less than 2 is to prevent clipped signals from being labeled
    #         no_motion = npall(accel_rsd < self.sd_crit, axis=1) & npall(abs(accel_rm) < 2, axis=1)
    #         # nans are automatically excluded
    #
    #         # trim to no motion only
    #         accel_rsd = accel_rsd[no_motion]
    #         accel_rm = accel_rm[no_motion]
    #         temp_rm = temp_rm[no_motion]
    #
    #         if accel_rsd.shape[0] > 1:
    #             # starting error
    #             cal_error_start = around(mean(abs(norm(accel_rsd, axis=1) - 1)), decimals=5)
    #
    #             # check if the sphere is well populated
    #             tel = (
    #                 (accel_rm.min(axis=0) < -self.sphere_crit)
    #                 & (accel_rm.max(axis=0) > self.sphere_crit)
    #             ).sum()
    #             if tel == 3:
    #                 sphere_populated = True
    #             else:
    #                 sphere_populated = False
    #         else:
    #             sphere_populated = False
    #
    #         offset = zeros(3)
    #         scale = ones(3)
    #         temp_offset = zeros((1, 3))
    #
    #         if not sphere_populated:
    #             warn(f"Recalibration not done with {self.min_hours + i_h * 12} hours due to "
    #                  f"insufficient non-movement data available")
    #         else:
    #             mean_temp = mean(temp_rm)
    #             in_acc = accel_rm
    #             in_temp = (temp_rm - mean_temp).reshape((-1, 1))
    #
    #             weights = ones(in_acc.shape[0])
    #             res = [Inf]
    #             LR = LinearRegression()
    #
    #             for niter in range(self.max_iter):
    #                 curr = (in_acc + offset) * scale + in_temp @ temp_offset
    #
    #                 closest_point = curr / sqrt(sum(curr**2, axis=1, keepdims=True))
    #                 offsetch = zeros(3)
    #                 scalech = ones(3)
    #                 toffch = zeros((1, 3))
    #
    #                 for k in range(3):
    #                     # there was some code dropping NANs from closest point, but these should
    #                     # be taken care of in the original mask. Division by zero should also
    #                     # not be happening during motionless data, where 1 value should always be close
    #                     # to 1
    #                     x_ = vstack((ones(curr.shape[0]), curr[:, k], in_temp[:, k])).T
    #                     LR.fit(
    #                         x_,
    #                         closest_point[:, k],
    #                         sample_weight=weights
    #                     )
    #                     offsetch[k] = LR.coef_[0]
    #                     scalech[k] = LR.coef_[1]
    #                     toffch[k] = LR.coef_[2]
    #                     curr[:, k] = x_ @ LR.coef_
    #
    #                 offset = offset + offsetch / (scale * scalech)
    #                 temp_offset = temp_offset * scalech + toffch
    #
    #                 scale = scale * scalech
    #                 res.append(
    #                     3 * mean(weights * (curr - closest_point)**2 / sum(weights))
    #                 )
    #                 weights = minimum(
    #                     1 / sqrt(sum((curr - closest_point)**2)),
    #                     100
    #                 )
    #                 if abs(res[niter] - res[niter - 1]) < self.tol:
    #                     break
    #
    #             in_acc = (in_acc + offset) * scale + (in_temp - mean_temp) * temp_offset
    #             cal_error_end = around(mean(abs(norm(in_acc, axis=1) - 1)), decimals=5)
    #
    #             # assess whether calibration error has been sufficiently improved
    #             if (cal_error_end < cal_error_start) and (cal_error_end < 0.01):
    #                 break
    #
    # def predict2(self, time=None, accel=None, *, apply=True, temp=None, **kwargs):
    #     """
    #     predict(time, accel, *, temp=None)
    #
    #     Parameters
    #     ----------
    #     time : numpy.ndarray
    #         (N, ) array of unix timestamps, in seconds.
    #     accel : numpy.ndarray
    #         (N, 3) array of accelerations, in units of 'g'.
    #     apply : bool, optional
    #         Apply the calibration to the accelerometer data. Default is True. If False, the values
    #         are only returned in `calibration_results` and acceleration is unchanged from the
    #         input.
    #     temp : numpy.ndarray, optional
    #         (N, ) array of temperatures. If not provided (None), no temperature based calibration is
    #         applied.
    #
    #     Returns
    #     -------
    #     calibration_results : dict
    #         The computed calibration parameters.
    #     data : dict
    #         Data that was passed in. Calibration applied to acceleration if `apply=True`.
    #     """
    #     super().predict(
    #         time=time, accel=accel, apply=apply, temp=temp, **kwargs
    #     )
    #
    #     fs = 1 / mean(diff(time[:500]))  # only need a rough estimate
    #     n10 = int(10 / fs)  # elements in 10 seconds
    #     nh = int(self.min_hours * 3600 / fs)  # elements in min_hours hours
    #     n12h = int(12 * 3600 / fs)  # elements in 12 hours
    #
    #     accel_rsd, accel_rm = rolling_sd(accel, n10, n10, axis=0, return_previous=True)
    #     mag_rm = rolling_mean(norm(accel, axis=1), n10, n10)
    #
    #     if temp is not None:
    #         temp_rm = rolling_mean(temp, n10, n10)
    #     else:
    #         temp_rm = zeros(mag_rm.size)
    #
    #     # less than 2 is to prevent clipped signals from being labeled
    #     no_motion = npall(accel_rsd < self.sd_crit, axis=1) & npall(abs(accel_rm) < 2, axis=1)
    #     # also add in nan values.  Only need one of accel_rm and accel_rsd
    #     no_motion |= npany(isnan(accel_rm)) & isnan(mag_rm) & isnan(temp_rm)
    #
    #     # trim to no motion only
    #     accel_rsd = accel_rsd[no_motion]
    #     accel_rm = accel_rm[no_motion]
    #     mag_rm = mag_rm[no_motion]
    #     temp_rm = temp_rm[no_motion]
    #
    #     if accel_rsd.shape[0] > 1:
    #         n_points = accel_rsd.shape[0]
    #
    #         # starting error
    #         cal_error_start = around(mean(abs(norm(accel_rsd, axis=1) - 1)), decimals=5)
    #
    #         # check if the sphere is well populated
    #         tel = (
    #                 (accel_rm.min(axis=0) < -self.sphere_crit)
    #                 & (accel_rm.max(axis=0) > self.sphere_crit)
    #         ).sum()
    #         if tel == 3:
    #             sphere_populated = True
    #         else:
    #             sphere_populated = False
    #             warn("Recalibration not done because no non-movement data available")
    #     else:
    #         warn(
    #             "Recalibration not done because not enough data in the file or because the file "
    #             "is corrupt"
    #         )
    #         sphere_populated = False
    #
    #     offset = zeros(3)
    #     scale = ones(3)
    #     temp_offset = zeros((1, 3))
    #
    #     if sphere_populated:
    #         mean_temp = mean(temp_rm)
    #         in_acc = accel_rm
    #         in_temp = (temp_rm - mean_temp).reshape((-1, 1))
    #
    #         weights = ones(accel_rm.shape[0])
    #         res = [Inf]
    #         LR = LinearRegression()
    #
    #         for iter in range(self.max_iter):
    #             curr = (in_acc + offset) * scale + in_temp @ temp_offset
    #
    #             closestpoint = curr / sqrt(sum(curr**2, axis=1, keepdims=True))
    #             offsetch = zeros(3)
    #             scalech = ones(3)
    #             toffch = zeros((1, 3))
    #
    #             for k in range(3):
    #                 # there was some code dropping NANs from closest point, but these should
    #                 # be taken care of in the original mask. Division by zero should also
    #                 # not be happening during motionless data, where 1 value should always be close
    #                 # to 1
    #                 x_ = vstack((ones(curr.shape[0]), curr[:, k], in_temp[:, k])).T
    #                 LR.fit(
    #                     x_,
    #                     closestpoint[:, k],
    #                     sample_weight=weights
    #                 )
    #                 offsetch[k] = LR.coef_[0]
    #                 scalech[k] = LR.coef_[1]
    #                 toffch[k] = LR.coef_[2]
    #                 curr[:, k] = x_ @ LR.coef_
    #
    #             offset = offset + offsetch / (scale * scalech)
    #             temp_offset = temp_offset * scalech + toffch
    #
    #             scale = scale * scalech
    #             res.append(
    #                 3 * mean(weights * (curr - closestpoint)**2 / sum(weights))
    #             )
    #             weights = minimum(
    #                 1 / sqrt(sum((curr - closestpoint)**2)),
    #                 1 / 0.01  # 100
    #             )
    #             if abs(res[iter] - res[iter-1]) < self.tol:
    #                 break
    #
    #         in_acc2 = (in_acc + offset) * scale + (in_temp - mean_temp) * temp_offset
    #         cal_error_end = around(mean(abs(norm(in_acc2, axis=1) - 1)), decimales=5)
    #
    #         # assess whether calibration error has been sufficiently improved
    #         if (cal_error_end < cal_error_start) and (cal_error_end < 0.01) and (nhoursused > self.min_hours):
    #             print("Calibration successful")
    #         else:
    #             add_more_hours()
    #
    #     # modify the acceleration with the scale/offset factors
    #     accel_corr = (accel + offset) * scale + (temp - mean_temp) * temp_offset
    #
    #     kwargs.update({self._time: time, self._acc: accel_corr, self._temp: temp})
    #
    #     return kwargs

    def do_iterative_closest_point_fit(self, acc, tmp, N, N10):
        """
        Perform the minimization using an iterative closest point fitting procedure

        Parameters
        ----------
        acc : numpy.ndarray
            Acceleration, shape (N, 3).
        tmp : numpy.ndarray
            Temperature, shape (N,).
        N : int
            Number of samples in `hours` worth of data.
        N10 : int
            Number of samples in 10 seconds worth of data.

        Returns
        -------
        finished : bool
            If the optimization finished succesfully.
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

        # compute the rolling SD and mean for the acceleration and temperature
        acc_rsd, acc_rm = rolling_sd(acc[:N], N10, N10, axis=0, return_previous=True)

        if tmp is not None:
            tmp_rm = rolling_mean(tmp[:N], N10, N10)
        else:
            tmp_rm = zeros(acc_rsd.shape[0])

        # get parts with no motion. <2 is to prevent clipped signals from being labeled
        no_motion = npall(acc_rsd < self.sd_crit, axis=1) & npall(abs(acc_rm) < 2, axis=1)
        # nans are automatically excluded

        # trim to no motion only
        acc_rsd = acc_rsd[no_motion]
        acc_rm = acc_rm[no_motion]
        tmp_rm = tmp_rm[no_motion]

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
                x_ = vstack((ones(curr.shape[0]), curr[:, k], tmp_rm[:, 0])).T
                LR.fit(x_, closest_point[:, k], sample_weight=weights)

                offsetch[k] = LR.coef_[0]
                scalech[k] = LR.coef_[1]
                toffch[0, k] = LR.coef_[2]
                curr[:, k] = x_ @ LR.coef_

            offset = offset + offsetch / (scale * scalech)
            tmp_scale = tmp_scale * scalech + toffch
            scale = scale * scalech

            res.append(
                3 * mean(weights[:, None] * (curr - closest_point)**2 / weights.sum())
            )
            weights = minimum(
                1 / norm(curr - closest_point, axis=1),
                100
            )

            # tolerance condition during iteration
            if abs(res[niter] - res[niter - 1]) < self.tol:
                break

        # end of iteration (break or end) calibration error assessment
        acc_rm = (acc_rm + offset) * scale + tmp_rm * tmp_scale
        cal_error_end = around(mean(abs(norm(acc_rm, axis=1) - 1)), decimals=5)

        # assess if calibration error has been significantly improved
        if (cal_error_end < cal_error_start) and (cal_error_end < 0.01):
            return True, offset, scale, tmp_scale, tmp_mean

