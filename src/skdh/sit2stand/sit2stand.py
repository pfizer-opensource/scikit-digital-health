"""
Sit-to-stand transfer detection and processing

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import (
    array,
    sum,
    mean,
    std,
    around,
    arange,
    nonzero,
    diff,
    ascontiguousarray,
)
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt, find_peaks
from pywt import cwt, scale2frequency

from skdh.base import BaseProcess
from skdh.sit2stand.detector import Detector, pad_moving_sd


class Sit2Stand(BaseProcess):
    """
    Sit-to-stand transfer detection and processing.

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
    long_still : float, optional
        Length of time of stillness for it to be considered a long period of stillness.
        Used to determine the integration window limits when available. Default is 0.5s
    still_window : float, optional
        Length of the moving window for calculating the moving statistics for
        determining stillness. Default is 0.3s.
    gravity_pass_order : int, optional
        Low-pass filter order for estimating the direction of gravity by low-pass
        filtering the raw acceleration. Default is 4.
    gravity_pass_cutoff : float, optional
        Low-pass filter frequency cutoff for estimating the direction of gravity.
        Default is 0.8Hz.
    continuous_wavelet : str, optional
        Continuous wavelet to use for signal deconstruction. Default is `gaus1`. CWT
        coefficients will be summed in the frequency range defined by `power_band`
    power_band : {array_like, int, float}, optional
        Frequency band in which to sum the CWT coefficients. Either an array_like of
        length 2, with the lower and upper limits, or a number, which will be taken as
        the upper limit, and the lower limit will be set to 0. Default is [0, 0.5].
    power_peak_kw : {None, dict}, optional
        Extra key-word arguments to pass to `scipy.signal.find_peaks` when finding
        peaks in the summed CWT coefficient power band data. Default is None, which
        will use the default parameters except setting minimum height to 90, unless
        `power_std_height` is True.
    power_std_height : bool, optional
        Use the standard deviation of the power for peak finding. Default is True.
        If True, the standard deviation height will overwrite the `height` setting in
        `power_peak_kw`.
    power_std_trim : float, int, optional
        Number of seconds to trim off the start and end of the power signal before
        computing the standard deviation for `power_std_height`. Default is 0s, which
        will not trim anything. Suggested value of trimming is 0.5s.
    lowpass_order : int, optional
        Initial low-pass filtering order. Default is 4.
    lowpass_cutoff : float, optional
        Initial low-pass filtering cuttoff, in Hz. Default is 5Hz.
    reconstruction_window : float, optional
        Window to use for moving average, in seconds. Default is 0.25s.
    day_window : array-like
        Two (2) element array-like of the base and period of the window to use for
        determining days. Default is (0, 24), which will look for days starting at
        midnight and lasting 24 hours. None removes any day-based windowing.

    Notes
    -----
    The default height threshold of 90 in `power_peak_kw` was determined on data
    sampled at 128Hz, and would likely need to be adjusted for different sampling
    frequencies. Especially if using a different sampling frequency, use of
    `power_std_height=True` is recommended.

    `stillness_constraint` determines whether or not a sit-to-stand transition is
    required to start and the end of a still period in the data. This constraint is
    suggested for at-home data. For processing clinic data, it is suggested to set this
    to `False`, especially if processing a task where sit-to-stands are repeated in
    rapid succession.

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

    def __init__(
        self,
        *,
        stillness_constraint=True,
        gravity=9.81,
        thresholds=None,
        long_still=0.5,
        still_window=0.3,
        gravity_pass_order=4,
        gravity_pass_cutoff=0.8,
        continuous_wavelet="gaus1",
        power_band=None,
        power_peak_kw=None,
        power_std_height=True,
        power_std_trim=0,
        lowpass_order=4,
        lowpass_cutoff=5,
        reconstruction_window=0.25,
        day_window=(0, 24),
    ):
        super().__init__(
            # kwarg saving
            stillness_constraint=stillness_constraint,
            gravity=gravity,
            thresholds=thresholds,
            long_still=long_still,
            still_window=still_window,
            gravity_pass_order=gravity_pass_order,
            gravity_pass_cutoff=gravity_pass_cutoff,
            continuous_wavelet=continuous_wavelet,
            power_band=power_band,
            power_peak_kw=power_peak_kw,
            power_std_height=power_std_height,
            power_std_trim=power_std_trim,
            lowpass_order=lowpass_order,
            lowpass_cutoff=lowpass_cutoff,
            reconstruction_window=reconstruction_window,
            day_window=day_window,
        )

        # FILTER PARAMETERS
        self.cwave = continuous_wavelet

        if power_band is None:
            self.power_start_f = 0
            self.power_end_f = 0.5
        elif isinstance(power_band, (int, float)):
            self.power_start_f = 0
            self.power_end_f = power_band
        else:
            self.power_start_f, self.power_end_f = power_band

        self.std_height = power_std_height
        self.std_trim = min(0, power_std_trim)

        if power_peak_kw is None:
            self.power_peak_kw = {"height": 90 / 9.81}  # convert for g
        else:
            self.power_peak_kw = power_peak_kw

        self.lp_ord = lowpass_order
        self.lp_cut = lowpass_cutoff
        self.rwindow = reconstruction_window

        # for transfer detection
        self.detector = Detector(
            stillness_constraint=stillness_constraint,
            gravity=gravity,
            thresholds=thresholds,
            gravity_pass_order=gravity_pass_order,
            gravity_pass_cutoff=gravity_pass_cutoff,
            long_still=long_still,
            still_window=still_window,
        )

        if day_window is None:
            self.day_key = (-1, -1)
        else:
            self.day_key = tuple(day_window)

    def predict(self, time=None, accel=None, **kwargs):
        """
        predict(time, accel, *, day_ends={})

        Predict the sit-to-stand transfers, and compute per-transition quantities

        Parameters
        ----------
        time : ndarray
            (N, ) array of timestamps (in seconds since 1970-1-1 00:00:00)
        accel : ndarray
            (N, 3) array of acceleration, with units of 'g'.
        day_ends : dict, optional
            Optional dictionary containing (N, 2) arrays of start and stop indices for invididual
            days. Dictionary keys are in the format "{base}, {period}". If not provided, or the
            key specified by `day_window` is not found, no day-based windowing will be done.
        """
        super().predict(
            expect_days=True, expect_wear=False, time=time, accel=accel, **kwargs
        )

        # FILTERING
        # ======================================================
        # compute the sampling period
        dt = mean(diff(time[:500]))

        # setup filter
        sos = butter(self.lp_ord, 2 * self.lp_cut * dt, btype="low", output="sos")

        # results storage
        sts = {
            "Date": [],
            "Day Number": [],
            "Time": [],
            "Hour": [],
            "STS Start": [],
            "STS End": [],
            "Duration": [],
            "Max. Accel.": [],
            "Min. Accel.": [],
            "SPARC": [],
            "Vertical Displacement": [],
            "Partial": [],
        }

        for iday, day_idx in enumerate(zip(*self.day_idx)):
            start, stop = day_idx

            # compute the magnitude of the acceleration
            m_acc = norm(accel[start:stop, :], axis=1)
            # filtered acceleration
            f_acc = ascontiguousarray(
                sosfiltfilt(sos, m_acc, padtype="odd", padlen=None)
            )

            # reconstructed acceleration
            n_window = int(around(self.rwindow / dt))
            r_acc, *_ = pad_moving_sd(f_acc, n_window, 1)

            # get the frequencies first to limit computation necessary
            freqs = scale2frequency(self.cwave, arange(1, 65)) / dt
            f_mask = (
                nonzero((freqs <= self.power_end_f) & (freqs >= self.power_start_f))[0]
                + 1
            )

            # CWT power peak detection
            coefs, freq = cwt(r_acc, f_mask, self.cwave, sampling_period=dt)

            # sum coefficients over the frequencies in the power band
            power = sum(coefs, axis=0)

            # find the peaks in the power data
            if self.std_height:
                trim = int(self.std_trim / dt)
                self.power_peak_kw["height"] = std(
                    power[trim:-trim] if trim != 0 else power, ddof=1
                )

            power_peaks, _ = find_peaks(power, **self.power_peak_kw)

            self.detector.predict(
                sts, dt, time[start:stop], accel[start:stop, :], f_acc, power_peaks
            )

            # fill out the day information
            sts["Day Number"].extend(
                [iday + 1] * (len(sts["Date"]) - len(sts["Day Number"]))
            )

        # get rid of the partial transitions
        partial = array(sts["Partial"])

        for k in [i for i in sts if i != "Partial"]:
            sts[k] = array(sts[k])[~partial] if partial != array([]) else array([])

        sts.pop("Partial")

        kwargs.update({self._time: time, self._acc: accel})

        return (kwargs, sts) if self._in_pipeline else sts
