"""
Sit-to-stand transfer detection and processing

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
import datetime

from numpy import (
    array,
    zeros,
    ceil,
    around,
    sum,
    abs,
    gradient,
    where,
    diff,
    insert,
    append,
    sign,
    median,
    arange,
)
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.integrate import cumtrapz

from skdh.utility import moving_sd
from skdh.utility.internal import rle
from skdh.features.lib import extensions


# utility methods
def pad_moving_sd(x, wlen, skip):
    """
    Compute the centered moving average and standard deviation.

    Parameters
    ----------
    x : numpy.ndarray
        Datat to take the moving average and st. dev. on.
    wlen : int
        Window size in number of samples.
    skip : int
        Window start skip in samples.

    Returns
    -------
    m_mean : numpy.ndarray
        Moving mean
    m_std : numpy.ndarray
        Moving standard deviation.
    pad : int
        Pading for the array.
    """
    m_mn = zeros(x.shape)
    m_sd = zeros(x.shape)

    wlen = max(wlen, 2)
    pad = int(ceil(wlen / 2))
    nr = x.shape[0] // skip - wlen + 1

    m_sd[pad : pad + nr], m_mn[pad : pad + nr] = moving_sd(
        x, wlen, skip, axis=0, return_previous=True
    )

    m_mn[:pad], m_mn[pad + nr :] = m_mn[pad], m_mn[-pad]
    m_sd[:pad], m_sd[pad + nr :] = m_sd[pad], m_sd[-pad]

    return m_mn, m_sd, pad


def get_stillness(filt_accel, dt, gravity, window, long_still_time, thresholds):
    """
    Stillness determination based on filtered acceleration magnitude and jerk magnitude.

    Parameters
    ----------
    filt_accel : numpy.ndarray
        1D array of filtered magnitude of acceleration data, units of m/s^2.
    dt : float
        Sampling time, in seconds,
    gravity : float
        Gravitational acceleration in m/s^2, as measured by the sensor during
        motionless periods.
    window : float
        Moving statistics window length, in seconds.
    long_still_time : float
        Minimum time for stillness to be classified as a long still period.
    thresholds : dict
        Dictionary of the 4 thresholds to be used - accel moving avg, accel moving std,
        jerk moving avg, and jerk moving std.
        Acceleration average thresholds should be for difference from gravitional
        acceleration.

    Returns
    -------
    still : numpy.ndarray
        (N, ) boolean array of stillness (True)
    starts : numpy.ndarray
        (Q, ) array of indices where still periods start. Includes index 0 if still[0]
        is True. Q < (N/2)
    stops : numpy.ndarray
        (Q, ) array of indices where still periods end. Includes index N-1 if still[-1]
        is True. Q < (N/2)
    long_starts : numpy.ndarray
        (P, ) array of indices where long still periods start. P <= Q.
    long_stops : numpy.ndarray
        (P, ) array of indices where long still periods stop.
    """
    # compute the sample window length from the time value
    n_window = max(int(around(window / dt)), 2)
    # compute acceleration moving stats. pad the output of the utility functions
    acc_rm, acc_rsd, _ = pad_moving_sd(filt_accel, n_window, 1)
    # compute the jerk
    jerk = gradient(filt_accel, dt, edge_order=2)
    # compute the jerk moving stats
    jerk_rm, jerk_rsd, _ = pad_moving_sd(jerk, n_window, 1)

    # create the stillness masks
    arm_mask = abs(acc_rm - gravity) < thresholds["accel moving avg"]
    arsd_mask = acc_rsd < thresholds["accel moving std"]
    jrm_mask = abs(jerk_rm) < thresholds["jerk moving avg"]
    jrsd_mask = jerk_rsd < thresholds["jerk moving std"]

    still = arm_mask & arsd_mask & jrm_mask & jrsd_mask
    lengths, starts, vals = rle(still.astype(int))

    starts = starts[vals == 1]
    stops = starts + lengths[vals == 1]

    still_dt = (stops - starts) * dt

    long_starts = starts[still_dt > long_still_time]
    long_stops = stops[still_dt > long_still_time]

    return still, starts, stops, long_starts, long_stops


class Detector:
    def __str__(self):
        return "Sit2StandTransferDetector"

    def __repr__(self):
        ret = "Detector("
        ret += f"stillness_constraint={self.stillness_constraint!r}, "
        ret += f"gravity={self.grav!r}, "
        ret += f"thresholds={self.thresh!r}, "
        ret += f"gravity_pass_order={self.grav_ord!r}, "
        ret += f"gravity_pass_cutoff={self.grav_cut!r}, "
        ret += f"long_still={self.long_still!r}, "
        ret += f"still_window={self.still_window!r})"
        return ret

    def __init__(
        self,
        stillness_constraint=True,
        gravity=9.81,
        thresholds=None,
        gravity_pass_order=4,
        gravity_pass_cutoff=0.8,
        long_still=0.5,
        still_window=0.3,
    ):
        """
        Method for detecting sit-to-stand transitions based on a series of heuristic
        signal processing rules.

        Parameters
        ----------
        stillness_constraint : bool, optional
            Whether or not to impose the stillness constraint on the detected
            transitions. Default is True.
        gravity : float, optional
            Value of gravitational acceleration measured by the accelerometer when
            still. Default is 9.81 m/s^2.
        thresholds : dict, optional
            A dictionary of thresholds to change for stillness detection and transition
            verification. See *Notes* for default values. Only values present will be
            used over the defaults.
        gravity_pass_order : int, optional
            Low-pass filter order for estimating the direction of gravity by low-pass
            filtering the raw acceleration. Default is 4.
        gravity_pass_cutoff : float, optional
            Low-pass filter frequency cutoff for estimating the direction of gravity.
            Default is 0.8Hz.
        long_still : float, optional
            Length of time of stillness for it to be considered a long period of
            stillness. Used to determine the integration window limits when available.
            Default is 0.5s
        still_window : float, optional
            Length of the moving window for calculating the moving statistics for
            determining stillness. Default is 0.3s.

        Notes
        -----
        `stillness_constraint` determines whether or not a sit-to-stand transition is
        required to start and the end of a still period in the data. This constraint is
        suggested for at-home data. For processing clinic data, it is suggested to set
        this to `False`, especially if processing a task where sit-to-stands are
        repeated in rapid succession.

        Default thresholds:
            - stand displacement: 0.125  :: min displacement for COM for a transfer (m)
            - displacement factor: 0.75  :: min factor * median displacement for a valid transfer
            - transition velocity: 0.2   :: min vertical velocity for a valid transfer (m/s)
            - duration factor: 10        :: max factor between 1st/2nd part duration of transfer
            - accel moving avg: 0.2      :: max moving avg accel to be considered still (m/s^2)
            - accel moving std: 0.1      :: max moving std accel to be considered still (m/s^2)
            - jerk moving avg: 2.5       :: max moving average jerk to be considered still (m/s^3)
            - jerk moving std: 3         :: max moving std jerk to be considered still (m/s^3)

        References
        ----------
        .. [1] L. Adamowicz et al., “Assessment of Sit-to-Stand Transfers during Daily
            Life Using an Accelerometer on the Lower Back,” Sensors, vol. 20, no. 22,
            Art. no. 22, Jan. 2020, doi: 10.3390/s20226618.
        """
        # set the default thresholds
        self._default_thresholds = {
            "stand displacement": 0.125,
            "displacement factor": 0.75,
            "transition velocity": 0.2,
            "duration factor": 10,
            "accel moving avg": 0.2,
            "accel moving std": 0.1,
            "jerk moving avg": 2.5,
            "jerk moving std": 3,
        }
        # assign attributes
        self.stillness_constraint = stillness_constraint
        self.grav = gravity

        self.thresh = {i: self._default_thresholds[i] for i in self._default_thresholds}
        if thresholds is not None:
            self.thresh.update(thresholds)

        self.grav_ord = gravity_pass_order
        self.grav_cut = gravity_pass_cutoff
        self.long_still = long_still
        self.still_window = still_window

    def predict(self, sts, dt, time, raw_accel, filt_accel, power_peaks):
        # convert accel to m/s^2 so that integrated values/thresholds are in m/s^2
        raw_acc = raw_accel * self.grav
        filt_acc = filt_accel * self.grav

        still, starts, stops, lstill_starts, lstill_stops = get_stillness(
            filt_acc, dt, self.grav, self.still_window, self.long_still, self.thresh
        )

        # estimate of vertical acceleration
        v_acc = self._get_vertical_accel(dt, raw_acc)

        # iterate over the power peaks (potential s2s time points)
        prev_int_start = -1  # keep track of integration regions
        prev_int_end = -1

        n_prev = len(sts["STS Start"])  # previous number of transitions

        for ppk in power_peaks:
            try:  # look for the preceding end of stillness
                end_still = self._get_end_still(time, stops, lstill_stops, ppk)
            except IndexError:
                continue
            # look for the next start of stillness
            start_still, still_at_end = self._get_start_still(
                dt, time, stops, lstill_starts, ppk
            )

            # INTEGRATE between the determined indices
            if (end_still < prev_int_start) or (start_still > prev_int_end):
                # original subtracted gravity, however given how this is integrated this makes no
                # difference to the end result
                v_vel, v_pos = self._integrate(
                    v_acc[end_still:start_still], dt, still_at_end
                )

                # save integration region limits -- avoid extra processing if possible
                prev_int_start = end_still
                prev_int_end = start_still

                # get zero crossings
                pos_zc = insert(where(diff(sign(v_vel)) > 0)[0] + 1, 0, 0) + end_still
                neg_zc = (
                    append(where(diff(sign(v_vel)) < 0)[0] + 1, v_vel.size - 1)
                    + end_still
                )

            # maker sure the velocity is high enough to indicate a peak
            if v_vel[ppk - prev_int_start] < self.thresh["transition velocity"]:
                continue

            # transition start
            sts_start = self._get_transfer_start(dt, ppk, end_still, pos_zc, stops)
            if sts_start is None:
                continue
            # transition end
            try:
                sts_end = neg_zc[neg_zc > ppk][0]
            # TODO add data for tests that could address this one
            except IndexError:  # pragma: no cover :: no data for this currently
                continue

            # Run quality checks and if they pass add the transition to the results
            valid_transfer, t_start_i, t_end_i = self._is_transfer_valid(
                sts, ppk, time, v_pos, sts_start, sts_end, prev_int_start
            )
            if not valid_transfer:
                continue

            # compute s2s features
            dur_ = time[sts_end] - time[sts_start]
            mx_ = filt_acc[sts_start:sts_end].max()
            mn_ = filt_acc[sts_start:sts_end].min()
            vdisp_ = v_pos[t_end_i] - v_pos[t_start_i]
            sal_ = extensions.SPARC(
                norm(raw_acc[sts_start:sts_end], axis=1),
                1 / dt,
                4,
                10.0,
                0.05,
            )

            dtime = datetime.datetime.utcfromtimestamp(time[sts_start])
            sts["Date"].append(dtime.strftime("%Y-%m-%d"))
            sts["Time"].append(dtime.strftime("%H:%M:%S.%f"))
            sts["Hour"].append(dtime.hour)

            sts["STS Start"].append(time[sts_start])
            sts["STS End"].append(time[sts_end])
            sts["Duration"].append(dur_)
            sts["Max. Accel."].append(mx_)
            sts["Min. Accel."].append(mn_)
            sts["SPARC"].append(sal_)
            sts["Vertical Displacement"].append(vdisp_)

        # check to ensure no partial transitions
        vdisp_ndarr = array(sts["Vertical Displacement"][n_prev:])
        sts["Partial"].extend(
            (
                vdisp_ndarr < (self.thresh["displacement factor"] * median(vdisp_ndarr))
            ).tolist()
        )

    def _get_vertical_accel(self, dt, accel):
        r"""
        Get an estimate of the vertical acceleration component.

        Parameters
        ----------
        dt : float
            Sampling period in seconds.
        accel : numpy.ndarray
            (N, 3) array of acceleration.

        Returns
        -------
        vert_acc : numpy.ndarray
            (N, ) array of the estimated acceleration in the vertical direction.

        Notes
        -----
        First, an estimate of the vertical axis is found by using a strict
        low-pass filter with a cutoff designed to only capture the direction
        of the gravity vector. For example a cutoff might be 0.5Hz. The vertical
        direction is then computed per:

        .. math:: \hat{v}_g(t) = \frac{filter(y_a(t))}{||filter(y_t(t))||_2}

        :math:`\hat{v}_g(t)` is the vertical (gravity) axis vector as a function of
        time and :math:`y_a(t)` is the measured acceleration as a function of time.

        The vertical component of acceleration can then be obtained as the
        dot-product of the vertical axis and the acceleration per

        .. math:: \bar{a}_g(t) = \hat{v}_g(t) \cdot y_a(t)
        """
        sos = butter(self.grav_ord, 2 * self.grav_cut * dt, btype="low", output="sos")
        v_g = sosfiltfilt(sos, accel, axis=0, padlen=0)
        v_g /= norm(v_g, axis=1, keepdims=True)

        # estimate of the vertical acceleration
        v_acc = sum(v_g * accel, axis=1)

        return v_acc

    @staticmethod
    def _integrate(vert_accel, dt, still_at_end):
        """
        Double integrate the acceleration along 1 axis to get velocity and position

        Parameters
        ----------
        vert_accel : numpy.ndarray
            (N, ) array of acceleration values to integrate
        dt : float
            Sampling time in seconds
        still_at_end : bool
            Whether or not the acceleration provided ends with a still period. Determines drift
            mitigation strategy.

        Returns
        -------
        vert_vel : numpy.ndarray
            (N, ) array of vertical velocity
        vert_pos : numpy.ndarray
            (N, ) array of vertical positions
        """
        x = arange(vert_accel.size)

        # integrate and drift mitigate
        if not still_at_end:
            vel = detrend(cumtrapz(vert_accel, dx=dt, initial=0))
            if abs(vel[0]) > 0.05:  # if too far away from 0
                vel -= vel[0]  # reset the beginning back to 0
        else:
            vel_dr = cumtrapz(vert_accel, dx=dt, initial=0)
            vel = vel_dr - (
                ((vel_dr[-1] - vel_dr[0]) / (x[-1] - x[0])) * x
            )  # no intercept

        # integrate velocity to get position
        pos = cumtrapz(vel, dx=dt, initial=0)

        return vel, pos

    def _get_end_still(self, time, still_stops, lstill_stops, peak):
        if self.stillness_constraint:
            end_still = still_stops[still_stops < peak][-1]
            if (time[peak] - time[end_still]) > 2:
                raise IndexError
        else:
            end_still = lstill_stops[lstill_stops < peak][-1]
            if (
                time[peak] - time[end_still]
            ) > 30:  # don't want to integrate too far out
                raise IndexError
        return end_still

    def _get_start_still(self, dt, time, still_starts, lstill_starts, peak):
        try:
            start_still = lstill_starts[lstill_starts > peak][0]
            if (time[start_still] - time[peak]) < 30:
                still_at_end = True
            else:
                # try to use a set time past the transition
                start_still = int(peak + (5 / dt))
                still_at_end = False

            return start_still, still_at_end
        except IndexError:
            start_still = int(peak + (5 / dt))
            still_at_end = False

        return start_still, still_at_end

    def _get_transfer_start(self, dt, peak, end_still, pos_zc, stops):
        if self.stillness_constraint:
            sts_start = end_still
        else:
            try:  # look for the previous positive zero crossing as the start
                sts_start = pos_zc[pos_zc < peak][-1]
                p_still = stops[stops < peak][-1]
                # possibly use the end of stillness if it is close enough
                if -0.5 < (dt * (p_still - sts_start)) < 0.7:
                    sts_start = p_still
            except IndexError:
                return None

        return sts_start

    def _is_transfer_valid(
        self, res, peak, time, vp, sts_start, sts_end, prev_int_start
    ):
        """
        Check if the sit-to-stand transfer is a valid one.

        Parameters
        ----------
        res : dict
            Dictionary of sit-to-stand transfers.
        peak : int
            Peak index.
        time : np.ndarray
            Timestamps in seconds.
        vp : numpy.ndarray
            Vertical position array.
        sts_start : int
            Sit-to-stand transfer start index.
        sts_end : int
            Sit-to-stand transfer end index.
        prev_int_start : int
            Index of the integration start.

        Returns
        -------
        valid_transition : bool
            If the transition is valid.
        """
        if len(res["STS Start"]) > 0:
            if (time[sts_start] - res["STS Start"][-1]) <= 0.4:
                return False, None, None

        # get the integrated value start index
        t_start_i = sts_start - prev_int_start
        t_end_i = sts_end - prev_int_start

        # check that the STS time is not too long
        qc1 = (time[sts_end] - time[sts_start]) < 4.5  # threshold from various lit

        # check that first part of the s2s is not too much longer than the second part
        dt_part_1 = time[peak] - time[sts_start]
        dt_part_2 = time[sts_end] - time[peak]
        qc2 = dt_part_1 < (self.thresh["duration factor"] * dt_part_2)

        # check that the start and end are not equal
        qc3 = t_start_i != t_end_i

        # check that there is enough displacement for an actual STS
        qc4 = (vp[t_end_i] - vp[t_start_i]) > self.thresh["stand displacement"]

        return qc1 & qc2 & qc3 & qc4, t_start_i, t_end_i
