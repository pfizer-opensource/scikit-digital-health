"""
Sit-to-stand transfer detection and processing

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import array, zeros, ceil, around, mean, std, sum, abs, gradient, where, diff, insert, \
    append, sign, median, arange, sqrt, log2
from numpy.fft import fft
from numpy.linalg import norm
from numpy.lib import stride_tricks
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.integrate import cumtrapz


# utility methods
def moving_stats(seq, window):
    """
    Compute the centered moving average and standard deviation.

    Parameters
    ----------
    seq : numpy.ndarray
        Data to take the moving average and standard deviation on.
    window : int
        Window size for the moving average/standard deviation.

    Returns
    -------
    m_mn : numpy.ndarray
        Moving average
    m_st : numpy.ndarray
        Moving standard deviation
    pad : int
        Padding at beginning of the moving average and standard deviation
    """
    def rolling_window(x, wind):
        if not x.flags['C_CONTIGUOUS']:  # pragma: no cover :: should never get here
            raise ValueError("Data must be C-contiguous in order to window for moving statistics")
        shape = x.shape[:-1] + (x.shape[-1] - wind + 1, wind)
        strides = x.strides + (x.strides[-1],)
        return stride_tricks.as_strided(x, shape=shape, strides=strides)

    if seq.ndim != 1:
        raise ValueError('seq must be 1D')
    assert seq.flags['C_CONTIGUOUS'], 'seq must be C-contiguous'  # just in case

    m_mn = zeros(seq.shape)
    m_st = zeros(seq.shape)

    if window < 2:
        window = 2

    pad = int(ceil(window / 2))
    rw_seq = rolling_window(seq, window)

    n = rw_seq.shape[0]

    m_mn[pad:pad + n] = mean(rw_seq, axis=-1)
    m_st[pad:pad + n] = std(rw_seq, axis=-1, ddof=1)

    m_mn[:pad], m_mn[pad + n:] = m_mn[pad], m_mn[-pad]
    m_st[:pad], m_st[pad + n:] = m_st[pad], m_st[-pad]
    return m_mn, m_st, pad


def get_stillness(filt_accel, dt, gravity, window, thresholds):
    """
    Stillness determination based on filtered acceleration magnitude and jerk magnitude

    Parameters
    ----------
    filt_accel : numpy.ndarray
        1D array of filtered magnitude of acceleration data, units of m/s^2
    dt : float
        Sampling time, in seconds
    gravity : float
        Gravitational acceleration in m/s^2, as measured by the sensor during motionless periods
    window : float
        Moving statistics window length, in seconds
    thresholds : dict
        Dictionary of the 4 thresholds to be used - accel moving avg, accel moving std,
        jerk moving avg, and jerk moving std.
        Acceleration average thresholds should be for difference from gravitional acceleration.

    Returns
    -------
    still : numpy.ndarray
        (N, ) boolean array of stillness (True)
    starts : numpy.ndarray
        (Q, ) array of indices where still periods start. Includes index 0 if still[0] is True.
        Q < (N/2)
    stops : numpy.ndarray
        (Q, ) array of indices where still periods end. Includes index N-1 if still[-1] is True.
        Q < (N/2)
    """
    # compute the sample window length from the time value
    n_window = int(around(window / dt))
    # compute acceleration moving stats
    acc_rm, acc_rsd, _ = moving_stats(filt_accel, n_window)
    # compute the jerk
    jerk = gradient(filt_accel, dt, edge_order=2)
    # compute the jerk moving stats
    jerk_rm, jerk_rsd, _ = moving_stats(jerk, n_window)

    # create the stillness masks
    arm_mask = abs(acc_rm - gravity) < thresholds['accel moving avg']
    arsd_mask = acc_rsd < thresholds['accel moving std']
    jrm_mask = abs(jerk_rm) < thresholds['jerk moving avg']
    jrsd_mask = jerk_rsd < thresholds['jerk moving std']

    still = arm_mask & arsd_mask & jrm_mask & jrsd_mask
    starts = where(diff(still.astype(int)) == 1)[0]
    stops = where(diff(still.astype(int)) == -1)[0]

    if still[0]:
        starts = insert(starts, 0, 0)
    if still[-1]:
        stops = append(stops, len(still) - 1)

    return still, starts, stops


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

    def __init__(self, stillness_constraint=True, gravity=9.81, thresholds=None,
                 gravity_pass_order=4, gravity_pass_cutoff=0.8, long_still=0.5, still_window=0.3):
        """
        Method for detecting sit-to-stand transitions based on a series of heuristic signal
        processing rules.

        Parameters
        ----------
        stillness_constraint : bool, optional
            Whether or not to impose the stillness constraint on the detected transitions.
            Default is True.
        gravity : float, optional
            Value of gravitational acceleration measured by the accelerometer when still.
            Default is 9.81 m/s^2.
        thresholds : dict, optional
            A dictionary of thresholds to change for stillness detection and transition
            verification. See *Notes* for default values. Only values present will be used
            over the defaults.
        gravity_pass_order : int, optional
            Low-pass filter order for estimating the direction of gravity by low-pass filtering
            the raw acceleration. Default is 4.
        gravity_pass_cutoff : float, optional
            Low-pass filter frequency cutoff for estimating the direction of gravity.
            Default is 0.8Hz.
        long_still : float, optional
            Length of time of stillness for it to be considered a long period of stillness.
            Used to determine the integration window limits when available. Default is 0.5s
        still_window : float, optional
            Length of the moving window for calculating the moving statistics for determining
            stillness. Default is 0.3s.

        Notes
        -----
        `stillness_constraint` determines whether or not a sit-to-stand transition is required to
        start and the end of a still period in the data. This constraint is suggested for at-home
        data. For processing clinic data, it is suggested to set this to `False`, especially if
        processing a task where sit-to-stands are repeated in rapid succession.

        Default thresholds:
            - stand displacement: 0.125  :: min displacement for COM for a transfer (m)
            - displacement factor: 0.75  :: min factor * median displacement for a valid transfer
            - transition velocity: 0.2   :: min vertical velocity for a valid transfer (m/s)
            - duration factor: 10        :: max factor between 1st/2nd part duration of transfer
            - accel moving avg: 0.2      :: max moving avg accel to be considered still (m/s^2)
            - accel moving std: 0.1      :: max moving std accel to be considered still (m/s^2)
            - jerk moving avg: 2.5       :: max moving average jerk to be considered still (m/s^3)
            - jerk moving std: 3         :: max moving std jerk to be considered still (m/s^3)

        """
        # set the default thresholds
        self._default_thresholds = {
            'stand displacement': 0.125,
            'displacement factor': 0.75,
            'transition velocity': 0.2,
            'duration factor': 10,
            'accel moving avg': 0.2,
            'accel moving std': 0.1,
            'jerk moving avg': 2.5,
            'jerk moving std': 3
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

        still, starts, stops = get_stillness(
            filt_acc, dt, self.grav, self.still_window, self.thresh
        )
        still_dt = (stops - starts) * dt

        lstill_starts = starts[still_dt > self.long_still]
        lstill_stops = stops[still_dt > self.long_still]

        # compute an estimate of the direction of gravity, assumed to be the vertical direction
        sos = butter(self.grav_ord, 2 * self.grav_cut * dt, btype='low', output='sos')
        vert = sosfiltfilt(sos, raw_acc, axis=0, padlen=0)
        vert /= norm(vert, axis=1, keepdims=True)

        # estimate of vertical acceleration
        v_acc = sum(vert * raw_acc, axis=1)

        # iterate over the power peaks (potentail s2s time points)
        prev_int_start = -1  # keep track of integration regions
        prev_int_end = -1

        n_prev = len(sts['STS Start'])  # previous number of transitions

        for ppk in power_peaks:
            try:  # look for the preceeding end of stillness
                end_still = self._get_end_still(time, stops, lstill_stops, ppk)
            except IndexError:
                continue
            try:  # look for the next start of stillness
                start_still, still_at_end = self._get_start_still(time, stops, lstill_starts, ppk)
            except IndexError:
                start_still = int(ppk + (5 / dt))  # try to use a set time past the transition
                still_at_end = False

            # INTEGRATE between the determined indices
            if (end_still < prev_int_start) or (start_still > prev_int_end):
                v_vel, v_pos = self._integrate(v_acc[end_still:start_still], dt, still_at_end)

                # save integration region limits -- avoid extra processing if possible
                prev_int_start = end_still
                prev_int_end = start_still

                # get zero crossings
                pos_zc = insert(where(diff(sign(v_vel)) > 0)[0], 0, 0) + end_still
                neg_zc = append(where(diff(sign(v_vel)) < 0)[0], v_vel.size-1) + end_still

            # maker sure the velocity is high enough to indicate a peak
            if v_vel[ppk - prev_int_start] < self.thresh['transition velocity']:
                continue

            # transition start
            if self.stillness_constraint:
                sts_start = end_still
            else:
                try:  # look for the previous positive zero crossing as the start
                    sts_start = pos_zc[pos_zc < ppk][-1]
                    p_still = stops[stops < ppk][-1]
                    # possibly use the end of stillness if it is close enough
                    if -0.5 < (dt * (p_still - sts_start)) < 0.7:
                        sts_start = p_still
                # TODO add data for tests that could address this one
                except IndexError:  # pragma: no cover :: no data for this currently
                    continue
            # transition end
            try:
                sts_end = neg_zc[neg_zc > ppk][0]
            # TODO add data for tests that could address this one
            except IndexError:  # pragma: no cover :: no data for this currently
                continue

            # QUALITY CHECKS
            # ==============
            t_start_i = sts_start - prev_int_start  # integrated value start index
            t_end_i = sts_end - prev_int_start

            # check that the STS time is not too long
            qc1 = (time[sts_end] - time[sts_start]) < 4.5  # threshold from various lit

            # check that the first half of the s2s is not too much longer than the second half
            dt_half_1 = time[ppk] - time[sts_start]
            dt_half_2 = time[sts_end] - time[ppk]
            qc2 = dt_half_1 < (self.thresh['duration factor'] * dt_half_2)

            # check that the start and end are not equal
            qc3 = t_start_i != t_end_i

            # check that there is enough displacement for an actual STS
            qc4 = (v_pos[t_end_i] - v_pos[t_start_i]) > self.thresh['stand displacement']

            if not (qc1 & qc2 & qc3 & qc4):  # if not all checks are passed :: pragma: no cover
                continue

            # sit to stand assignment
            if len(sts['STS Start']) == 0:
                # compute s2s features
                dur_ = time[sts_end] - time[sts_start]
                mx_ = filt_acc[sts_start:sts_end].max()
                mn_ = filt_acc[sts_start:sts_end].min()
                vdisp_ = v_pos[t_end_i] - v_pos[t_start_i]
                sal_ = self.sparc(norm(raw_acc[sts_start:sts_end], axis=1), 1 / dt)

                sts['STS Start'].append(time[sts_start])
                sts['STS End'].append(time[sts_end])
                sts['Duration'].append(dur_)
                sts['Max. Accel.'].append(mx_)
                sts['Min. Accel.'].append(mn_)
                sts['SPARC'].append(sal_)
                sts['Vertical Displacement'].append(vdisp_)
            else:
                if (time[sts_start] - sts['STS Start'][-1]) > 0.4:
                    # compute s2s features
                    dur_ = time[sts_end] - time[sts_start]
                    mx_ = filt_acc[sts_start:sts_end].max()
                    mn_ = filt_acc[sts_start:sts_end].min()
                    vdisp_ = v_pos[t_end_i] - v_pos[t_start_i]
                    sal_ = self.sparc(norm(raw_acc[sts_start:sts_end], axis=1), 1 / dt)

                    sts['STS Start'].append(time[sts_start])
                    sts['STS End'].append(time[sts_end])
                    sts['Duration'].append(dur_)
                    sts['Max. Accel.'].append(mx_)
                    sts['Min. Accel.'].append(mn_)
                    sts['SPARC'].append(sal_)
                    sts['Vertical Displacement'].append(vdisp_)

        # check to ensure no partial transitions
        vdisp_ndarr = array(sts['Vertical Displacement'][n_prev:])
        sts['Partial'].extend(
            (vdisp_ndarr < (self.thresh['displacement factor'] * median(vdisp_ndarr))).tolist()
        )

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
            vel = vel_dr - (((vel_dr[-1] - vel_dr[0]) / (x[-1] - x[0])) * x)  # no intercept

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
            if (time[peak] - time[end_still]) > 30:  # don't want to integrate too far out
                raise IndexError
        return end_still

    def _get_start_still(self, time, still_starts, lstill_starts, peak):
        still_at_end = False
        start_still = lstill_starts[lstill_starts > peak][0]
        if (time[start_still] - time[peak]) < 30:
            still_at_end = True
        else:
            raise IndexError
        return start_still, still_at_end

    @staticmethod
    def sparc(x, fs):
        """
        SPectral ARC length metric for quantifying smoothness

        Parameters
        ----------
        x : numpy.ndarray
            Array containing the data to be analyzed for smoothness
        fs : float
            Sampling frequency

        Returns
        -------
        sal : float
            The spectral arc length estimate of the given data's smoothness
        (f, Mf) : (numpy.ndarray, numpy.ndarray)
            The frequency and the magnitude spectrum of the input data. This spectral is from 0
            to the Nyquist frequency
        (f_sel, Mf_sel) : (numpy.ndarray, numpy.ndarray)
            The portion of the spectrum that is selected for calculating the spectral arc length

        References
        ----------
        S. Balasubramanian, A. Melendez-Calderon, A. Roby-Brami, E. Burdet. "On the analysis of
        movement smoothness." Journal of NeuroEngineering and Rehabilitation. 2015.
        """
        padlevel, fc, amp_th = 4, 10.0, 0.05

        # number of zeros to be padded
        nfft = int(2**(ceil(log2(len(x))) + padlevel))

        # frequency
        f = arange(0, fs, fs / nfft)
        # normalized magnitude spectrum
        Mf = abs(fft(x, nfft))
        Mf = Mf / Mf.max()

        # indices to choose only the spectrum withing the given cutoff frequency Fc
        # NOTE: this is a low pass filtering operation to get rid of high frequency noise from
        # affecting the next step (amplitude threshold based cutoff for arc length calculation
        fc_inx = ((f <= fc) * 1).nonzero()
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]

        # choose the amplitude threshold based cutoff frequency. Index of the last point on the
        # magnitude spectrum is greater than or equal to the amplitude threshold
        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        fc_inx = arange(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        # calculate the arc length
        sal = -sum(sqrt((diff(f_sel) / (f_sel[-1] - f_sel[0]))**2 + diff(Mf_sel)**2))

        return sal
