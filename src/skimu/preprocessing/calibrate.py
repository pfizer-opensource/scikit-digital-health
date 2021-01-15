"""
Inertial data/sensor calibration

Lukas Adamowicz
Pfizer DMTI 2021
"""
from numpy import mean, diff

from skimu.base import _BaseProcess


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
    def __init__(self, sphere_crit=0.3, min_hours=72):
        if min_hours % 12 != 0 or min_hours <= 0:
            min_hours = ((min_hours // 12) + 1) * 12

        super().__init__(sphere_crit=sphere_crit, min_hours=min_hours)

        self.sphere_crit = sphere_crit
        self.min_hours = min_hours

    def predict(self, time=None, accel=None, *, apply=True, temp=None, **kwargs):
        """
        predict(time, accel, *, temp=None)

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        accel : numpy.ndarray
            (N, 3) array of accelerations, in units of 'g'.
        apply : bool, optional
            Apply the calibration to the accelerometer data. Default is True. If False, the values
            are only returned in `calibration_results` and acceleration is unchanged from the
            input.
        temp : numpy.ndarray, optional
            (N, ) array of temperatures. If not provided (None), no temperature based calibration is
            applied.

        Returns
        -------
        calibration_results : dict
            The computed calibration parameters.
        data : dict
            Data that was passed in. Calibration applied to acceleration if `apply=True`.
        """
        super().predict(
            time=time, accel=accel, apply=apply, temp=temp, **kwargs
        )

        fs = 1 / mean(diff(time[:500]))  # only need a rough estimate
        n10 = int(10 / fs)  # elements in 10 seconds
        nh = int(self.min_hours * 3600 / fs)  # elements in min_hours hours
        n12h = int(12 * 3600 / fs)  # elements in 12 hours

