"""
Gait event (initial/final contact) estimation

Lukas Adamowicz
Copyright 2023, Pfizer Inc. All rights reserved
"""

from warnings import warn

from numpy import std, argmax, diff, sign, array, int_
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
from pywt import cwt, frequency2scale

from skdh.base import BaseProcess, handle_process_returns


class VerticalCwtGaitEvents(BaseProcess):
    """
    Predict initial and final contact events using a Continuous Wavelet Transform
    on the vertical acceleration.

    Parameters
    ----------
    use_cwt_scale_relation : bool, optional
        Use the optimal scale/frequency relationship (see Notes). This changes which
        scale is used for the smoothing/differentiation operation performed with the
        continuous wavelet transform. Default is True. See Notes for a caveat of the
        relationship.
    wavelet_scale : {"default", float, int}, optional
        The wavelet scale to use. If `use_cwt_scale_relation=True`, then this is only
        used initially to determine the optimal scale. If `False`, then is used as the
        scale for the initial and final contact event detection. `"default"`
        corresponds to the default scale from [1]_, scaled for the sampling frequency.
        If a float, this is the value in Hz that the desired wavelet decomposition
        happens. For reference, [1]_ used a frequency of 1.25Hz. If an integer,
        uses that value as the scale.
    round_scale : bool, optional
        Round the CWT scales to integers. Default is False.

    Notes
    -----
    The optimal scale/frequency relationship found in [2]_ was based on a cohort
    of only young women students. While it is recommended to use this relationship,
    the user should be aware of this shortfall in the generation of the relationship.

    References
    ----------
    .. [1] J. McCamley, M. Donati, E. Grimpampi, and C. Mazzà, “An enhanced estimate of initial
        contact and final contact instants of time using lower trunk inertial sensor data,”
        Gait & Posture, vol. 36, no. 2, pp. 316–318, Jun. 2012,
        doi: 10.1016/j.gaitpost.2012.02.019.
    .. [2] C. Caramia, C. De Marchis, and M. Schmid, “Optimizing the Scale of a Wavelet-Based
        Method for the Detection of Gait Events from a Waist-Mounted Accelerometer under
        Different Walking Speeds,” Sensors, vol. 19, no. 8, p. 1869, Jan. 2019,
        doi: 10.3390/s19081869.
    """

    def __init__(
        self,
        use_cwt_scale_relation=True,
        wavelet_scale="default",
        round_scale=False,
    ):
        super().__init__(
            use_cwt_scale_relation=use_cwt_scale_relation,
            wavelet_scale=wavelet_scale,
        )

        self.use_cwt_scale_relation = use_cwt_scale_relation
        self.wavelet_scale = wavelet_scale
        self.round = round_scale

    def handle_wavelet_scale(self, original_scale, fs):
        if self.wavelet_scale == "default":
            scale = original_scale
        elif isinstance(self.wavelet_scale, float):
            scale = max(0.2 / (self.wavelet_scale / fs), 1)
        elif isinstance(self.wavelet_scale, int):
            scale = self.wavelet_scale
        else:
            raise ValueError("Type or value of `wavelet_scale` not understood")
        return scale

    def get_cwt_scales(self, v_velocity, fs, mean_step_freq):
        """
        Get the CWT scales for the IC and FC events.

        Parameters
        ----------
        v_velocity : numpy.ndarray
            Vertical velocity, in units of "g*sec".
        fs : float
            Sampling frequency, in Hz.
        mean_step_freq : float, None
            Mean step frequency for this walking bout.

        Returns
        -------
        scale1 : int
            First scale for the CWT. For initial contact events.
        scale2 : int
            Second scale for the CWT. For final contact events.
        """
        # compute the original scale:
        # 1.25 comes from original paper, corresponds to desired frequency
        # 0.2 is the central frequency of the 'gaus1' wavelet (normalized to 1)
        original_scale = max(0.2 / (1.25 / fs), 1)

        scale = self.handle_wavelet_scale(original_scale, fs)

        if mean_step_freq is None:
            warn(
                "`mean_step_freq` has not been calculated. Unable to adjust wavelet "
                "scale based on step frequency.",
                UserWarning,
            )
            return scale, scale

        if self.use_cwt_scale_relation:
            # IC scale: -10 * sf + 56
            # FC scale: -52 * sf + 131
            # TODO: verify the FC scale equation. This is not in the paper (typo?) but
            # but is a guess from the graph
            # original fs was 250 hz, hence the conversion
            scale1 = (-10 * mean_step_freq + 56) * (fs / 250.0)
            scale2 = (-52 * mean_step_freq + 131) * (fs / 250.0)
            if self.round:
                scale1 = round(scale1)
                scale2 = round(scale2)
            # scale is range between 1 and 90
            scale1 = min(max(scale1, 1), 90)
            scale2 = min(max(scale2, 1), 90)
        else:
            scale1 = scale2 = scale

        return scale1, scale2

    @handle_process_returns(results_to_kwargs=True)
    def predict(
        self,
        time=None,
        accel=None,
        accel_filt=None,
        v_axis=None,
        v_axis_sign=None,
        *,
        fs=None,
        mean_step_freq=None,
        **kwargs,
    ):
        """
        predict(time, accel, accel_filt, v_axis, v_axis_sign, *, fs=None, mean_step_freq=None)

        Parameters
        ----------
        time
        accel
        accel_filt
        v_axis
        v_axis_sign
        fs
        mean_step_freq

        Returns
        -------
        results : dict
            Dictionary of the results, with the following items that can be used
            as inputs to downstream processing steps:
            - `initial_contacts`: detected initial contact events (heel-strikes).
            - `final_contacts`: detected final contact events (toe-offs).
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            accel_filt=accel_filt,
            v_axis=v_axis,
            v_axis_sign=v_axis_sign,
            fs=fs,
            **kwargs,
        )

        # integrate the vertical acceleration to get velocity
        vert_velocity = cumulative_trapezoid(
            accel_filt[:, v_axis], dx=1 / fs, initial=0
        )

        # get the CWT scales
        scale1, scale2 = self.get_cwt_scales(vert_velocity, fs, mean_step_freq)

        coef1, _ = cwt(vert_velocity, [scale1, scale2], "gaus1")
        """
        Find the local minima in the signal. This should technically always require using
        the negative signal in "find_peaks", however the way PyWavelets computes the
        CWT results in the opposite signal that we want.
        Therefore, if the sign of the acceleration was negative, we need to use the
        positve coefficient signal, and opposite for positive acceleration reading.
        """
        init_contact, *_ = find_peaks(
            -v_axis_sign * coef1[0], height=0.5 * std(coef1[0])
        )

        coef2, _ = cwt(coef1[1], scale2, "gaus1")
        """
        Peaks are the final contact points
        Same issue as above
        """
        final_contact, *_ = find_peaks(
            -v_axis_sign * coef2[0], height=0.5 * std(coef2[0])
        )

        res = {
            "initial_contacts": init_contact.astype(int_),
            "final_contacts": final_contact.astype(int_),
        }

        return res


class ApCwtGaitEvents(BaseProcess):
    """
    Predict gait events from a lumbar sensor based on AP acceleration and using
    a Continuous Wavelet Transform to smooth the raw signal.

    Parameters
    ----------
    ic_prom_factor : float, optional
        Factor multiplied by the standard deviation of the CWT coefficients to
        obtain a minimum prominence for IC peak detection. Default is 0.6.
    ic_dist_factor : float, optional
        Factor multiplying the mean step samples to obtain a minimum distance
        (in # of samples) between IC peaks. Default is 0.5.
    fc_prom_factor : float, optional
        Factor multiplying the standard deviation of the CWT coefficients to
        obtain a minimum prominence for FC peak detection. Default is 0.6
    fc_dist_factor : float, optional
        Factor multiplying the mean step samples to obtain a minimum distance
        (in # of samples) between FC peaks. Default is 0.6.
    """

    def __init__(
        self,
        ic_prom_factor=0.6,
        ic_dist_factor=0.5,
        fc_prom_factor=0.6,
        fc_dist_factor=0.6,
    ):
        super().__init__(
            ic_prom_factor=ic_prom_factor,
            ic_dist_factor=ic_dist_factor,
            fc_prom_factor=fc_prom_factor,
            fc_dist_factor=fc_dist_factor,
        )

        self.ic_pf = ic_prom_factor
        self.ic_df = ic_dist_factor
        self.fc_pf = fc_prom_factor
        self.fc_df = fc_dist_factor

    @handle_process_returns(results_to_kwargs=True)
    def predict(
        self,
        time=None,
        accel=None,
        accel_filt=None,
        ap_axis=None,
        ap_axis_sign=None,
        mean_step_freq=None,
        *,
        fs=None,
        **kwargs,
    ):
        """
        predict(time, accel, ap_axis, ap_axis_sign, mean_step_freq, *, fs=None)

        Parameters
        ----------
        time
        accel
        accel_filt
        ap_axis
        ap_axis_sign
        mean_step_freq
        fs
        kwargs

        Returns
        -------
        results : dict
            Dictionary of the results, with the following items that can be used
            as inputs to downstream processing steps:

            - `initial_contacts`: detected initial contact events (heel-strikes).
            - `final_contacts`: detected final contact events (toe-offs).
        """
        # compute the estimates for the scales
        f_cwt_ic = 1.3 * mean_step_freq - 0.3
        f_cwt_fc = 1.17 * mean_step_freq - 0.3
        scale_ic = frequency2scale("gaus1", f_cwt_ic / fs)
        scale_fc = frequency2scale("gaus1", f_cwt_fc / fs)

        # FINAL CONTACTS
        ap_vel = cumulative_trapezoid(accel_filt[:, ap_axis], dx=1 / fs, initial=0)
        coef_fc, _ = cwt(ap_vel, scale_fc, "gaus1")
        fcs, _ = find_peaks(
            ap_axis_sign * coef_fc[0],
            prominence=self.fc_pf * std(coef_fc[0]),
            distance=max(int(self.fc_df * fs / mean_step_freq), 1),
        )

        # INITIAL CONTACT
        coef_ic, _ = cwt(accel_filt[:, ap_axis], scale_ic, "gaus1")
        # get the peaks
        pks, _ = find_peaks(
            -ap_axis_sign * coef_ic[0],
            prominence=self.ic_pf * std(coef_ic[0]),
            distance=max(int(self.ic_df * fs / mean_step_freq), 1),
        )
        # minima/negative peaks
        npks, _ = find_peaks(
            ap_axis_sign * coef_ic[0],
            prominence=self.ic_pf * std(coef_ic[0]),
            distance=max(int(self.ic_df * fs / mean_step_freq), 1),
        )

        ics = []
        for fc in fcs:
            try:
                # lax time period, this wll be cleaned up by QC steps
                prev_npk = npks[(npks < fc) & (npks > (fc - 2 * fs))][-1]
                prev_pk = pks[pks < prev_npk][-1]
            except IndexError:
                continue

            # get the zero crossing
            zc = argmax(diff(sign(-ap_axis_sign * coef_ic[0][prev_pk:])) < 0)

            # i1 = prev_pk + zc
            # i2 = prev_npk
            # delta = (i2 - i1) / 2
            # ic = i1 + delta
            # ic = i1 + (i2 - i1) / 2
            # ic = prev_pk + zc + (prev_npk - prev_pk - zc) / 2
            # ic = (2prev_pk + 2zc + prev_npk - prev_pk - zc) / 2
            ic = int((prev_pk + zc + prev_npk) / 2)

            ics.append(ic)

        res = {
            "initial_contacts": array(ics).astype(int_),
            "final_contacts": fcs.astype(int_),
        }

        return res
