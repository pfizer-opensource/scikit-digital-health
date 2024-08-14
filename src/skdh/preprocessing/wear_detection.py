"""
Wear detection algorithms

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.

DETACH algorithm: Copyright (c) 2022 Adam Vert
"""

from warnings import warn

from numpy import (
    mean,
    diff,
    sum,
    roll,
    insert,
    append,
    nonzero,
    delete,
    concatenate,
    int_,
    full,
    sort,
    unique,
    isclose,
    asarray,
    ascontiguousarray,
)
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility import moving_mean, moving_sd, moving_max, moving_min
from skdh.utility.internal import rle, invert_indices
from skdh.utility.activity_counts import get_activity_counts


class DETACH(BaseProcess):
    r"""
    DEvice Temperature and Acceleration CHange algorithm for detecting wear/non-wear.

    Parameters
    ----------
    sd_thresh : float, optional
        Standard deviation threshold for an acceleration axis to trigger non-wear.
        Default is 0.008 g (8 milli-g).
    low_temperature_threshold : float, optional
        Low temperature threshold for non-wear classification. Default is 26.0 deg C.
    high_temperature_threshold : float, optional
        High temperature threshold for non-wear classification. Default is 30.0 deg C.
    decrease_threshold : float, optional.
        Temperature decrease rate-of-change threshold for non-wear classification.
        Default is -0.2.
    increase_threshold : float, optional
        Temperature increase rate-of-change threshold for non-wear classification.
        Default is 0.1
    n_axes_threshold : {1, 2, 3}
        Number of axes that must be below `sd_thresh` to be considered non-wear.
        Default is 2.
    window_size : {int, 'scaled', 'original'}, optional
        Window size in seconds, 'original', or 'scaled'. 'Original' uses the
        original 4-second long windows from [1]_, which was developed at a sampling
        frequency of 75hz for GeneActiv devices. 'scaled' uses the same principal
        as in [1]_ but will scale the window length to match the input sampling
        frequency (by a factor of 300, which is the block size for GeneActiv devices).
        Maximum if providing an integer is 15 seconds. Default is a window size
        of 1 second.

    References
    ----------
    .. [1] A. Vert et al., “Detecting accelerometer non-wear periods using change
        in acceleration combined with rate-of-change in temperature,” BMC Medical
        Research Methodology, vol. 22, no. 1, p. 147, May 2022, doi: 10.1186/s12874-022-01633-6.

    Notes
    -----
    This algorithm was based on work done with GENEActiv devices. While efforts
    were made to keep the algorithm device-agnostic, this should be kept in mind
    when deploying in alternative devices.

    Copyright (c) 2022 Adam Vert. Implementation here courtesy of release under
    an MIT license. The original implementation, and license can be found on
    `GitHub <https://github.com/nimbal/vertdetach>`.
    """

    def __init__(
        self,
        sd_thresh=0.008,
        low_temperature_threshold=26.0,
        high_temperature_threshold=30.0,
        decrease_threshold=-0.2,
        increase_threshold=0.1,
        n_axes_threshold=2,
        window_size=1,
    ):
        if n_axes_threshold not in [1, 2, 3]:
            n_axes_threshold = max(min(n_axes_threshold, 3), 1)
            warn(
                f"n_axes_threshold must be in {1, 2, 3}. Setting to {n_axes_threshold}",
                UserWarning,
            )

        if not isinstance(window_size, int) and window_size not in [
            "scaled",
            "original",
        ]:
            raise ValueError("`window_size` must be an int, or 'scaled' or 'original'")
        if isinstance(window_size, int):
            if window_size > 15:
                warn("`window_size` above 15 seconds. Setting to 15 seconds")
                window_size = 15

        super().__init__(
            sd_thresh=sd_thresh,
            low_temperature_threshold=low_temperature_threshold,
            high_temperature_threshold=high_temperature_threshold,
            decrease_threshold=decrease_threshold,
            increase_threshold=increase_threshold,
            n_axes_threshold=n_axes_threshold,
            window_size=window_size,
        )

        self.sd_thresh = sd_thresh
        self.low_temp = low_temperature_threshold
        self.high_temp = high_temperature_threshold
        self.decr_thresh = decrease_threshold
        self.incr_thresh = increase_threshold
        self.n_ax = n_axes_threshold
        self.wsize = window_size

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, time=None, accel=None, temperature=None, *, fs=None, **kwargs):
        """
        Detect periods of non-wear.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps (in seconds) since 1970-01-01.
        accel : numpy.ndarray
            (N, 3) array of measured acceleration values in units of g.
        temperature : numpy.ndarray
            (N,) array of measured temperature values during recording in deg C.
        fs : float, optional
            Sampling frequency, in Hz. If not provided, will be computed from
            `time`.

        Returns
        -------
        results : dict
            Dictionary of inputs, plus the key `wear` which is an array-like (N, 2)
            indicating the start and stop indices of wear.
        """
        # this implementation was aided by code released by the authors:
        # https://github.com/nimbal/vertdetach
        """
        MIT License

        Copyright (c) 2022 Adam Vert
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            temperature=temperature,
            fs=fs,
            **kwargs,
        )
        # dont start at 0 due to timestamp weirdness with some devices
        fs = 1 / mean(diff(time[1000:5000])) if fs is None else fs

        if isinstance(self.wsize, (int, float)):
            wsize = int(self.wsize)
        elif self.wsize.lower() == "scaled":
            wsize = int(300 / fs)
        elif self.wsize.lower() == "original":
            wsize = 4
        else:
            raise ValueError("Specified `window_size` in initialization not understood")

        # calculate default window lengths
        wlen_ds = int(fs * wsize)
        fs_ds = 1 / wsize
        wlen = int(fs * 60)  # at original sampling frequency
        wlen_5min = int(fs * 60 * 5)  # original sampling frequency
        wlen_ds_5min = int(fs_ds * 60 * 5)

        # compute a minute-long rolling standard deviation
        # this can be both forward and backwards looking with the correct indexing
        # current implementation matches the "forwards" looking from pandas
        rsd_acc = moving_sd(accel, wlen, 1, axis=0, trim=False, return_previous=False)
        # to get "backwards" looking, roll by `wlen - 1`

        # In the original algorithm they keep one temperature sample per GENEActiv
        # block (300 samples), resulting in a temperature sampling rate of 0.25hz
        # (accel fs=75). Here, temperature is expected to have the same sampling
        # frequency as the acceleration (ie temperature values are duplicated)

        # in theory, this will make a difference in the end result, however in practice
        # it does not seem to make a significant enough difference

        # "down-sample" the temperature by taking a moving mean. If using `scaled` for
        # window-size this would bring back the 1 temperature value per block, but it
        # should also handle data coming from other devices better
        temp_ds = moving_mean(temperature, wlen_ds, wlen_ds, trim=True)

        # filter the temperature data. make sure to use down-sampled frequency
        sos = butter(2, 2 * 0.005 / fs_ds, btype="low", output="sos")
        temp_f = ascontiguousarray(sosfiltfilt(sos, temp_ds))

        # convert to a slope (per minute). make sure to use down-sampled frequency
        delta_temp_f = diff(temp_f, prepend=1) * 60 * fs_ds

        # get the number of axes that are less than std_thresh, both forwards and backwards
        n_ax_under_sd_range_fwd = sum(rsd_acc < self.sd_thresh, axis=1)
        n_ax_under_sd_range_bwd = roll(n_ax_under_sd_range_fwd, wlen - 1)

        # find spots where at least N axes are below the StD threshold for at
        # least 90% of the next 5 minutes  (90% criteria below)
        perc_under_sd_range_5min_fwd = moving_mean(
            n_ax_under_sd_range_fwd
            >= self.n_ax,  # if more than N axes under StD Thresh
            wlen_5min,
            wlen_ds,
            trim=False,
        )
        perc_under_sd_range_5min_bwd = moving_mean(
            n_ax_under_sd_range_bwd
            >= self.n_ax,  # if more than N axes under StD Thresh
            wlen_5min,
            wlen_ds,
            trim=False,
        )

        # match the number of points between accel and temperature by down-sampling
        # accel to temperature
        # rsd_acc not used anymore
        n_ax_under_sd_range_fwd = n_ax_under_sd_range_fwd[::wlen_ds][: temp_ds.size]
        n_ax_under_sd_range_bwd = n_ax_under_sd_range_bwd[::wlen_ds][: temp_ds.size]
        perc_under_sd_range_5min_fwd = perc_under_sd_range_5min_fwd[: temp_ds.size]
        perc_under_sd_range_5min_bwd = perc_under_sd_range_5min_bwd[: temp_ds.size]

        # Get the maximum & minimum temperature in 5 minute windows
        max_temp_5min = moving_max(temp_f, wlen_ds_5min, 1, trim=False)
        min_temp_5min = moving_min(temp_f, wlen_ds_5min, 1, trim=False)

        # get the average temperature change in the next 5 minutes
        avg_temp_delta_5min = moving_mean(delta_temp_f, wlen_ds_5min, 1, trim=False)

        # get candidate non-wear start times. 90% criteria next 5 minutes comes here
        candidate_nw_starts = nonzero(
            (n_ax_under_sd_range_fwd >= self.n_ax)
            & (perc_under_sd_range_5min_fwd >= 0.9)
        )[0]

        # create the arrays of possible non-wear bout endings
        # to get a "backwards" looking window, take the moving windows, and add
        # wlen - 1 to the index

        # criteria 1: Rate of Change
        stops1 = nonzero(
            (n_ax_under_sd_range_bwd == 0)
            & (perc_under_sd_range_5min_bwd <= 0.50)
            & (avg_temp_delta_5min > self.incr_thresh)
        )[0]

        # criteria 2: absolute temperature
        stops2 = nonzero(
            (n_ax_under_sd_range_bwd == 0)
            & (perc_under_sd_range_5min_bwd <= 0.50)
            & (min_temp_5min > self.low_temp)
        )[0]

        candidate_nw_stops = sort(unique(concatenate((stops1, stops2))))

        # loop through the starts to identify valid starts, and find their stops
        prev_end = 0  # keep track of the last bout
        nonwear_starts = []  # store nonwear bout starts and stops
        nonwear_stops = []
        for start in candidate_nw_starts:
            if start < prev_end:
                continue  # skip because we are already past the current start

            valid_start = False
            end_initial = start + int(fs_ds * 60 * 5)  # add 5 minutes to start

            # start criteria 1: rate of change of temperature
            if (max_temp_5min[start] < self.high_temp) & (
                avg_temp_delta_5min[start] < self.decr_thresh
            ):
                valid_start = True

            # start criteria 2: absolute temperature path
            elif max_temp_5min[start] < self.low_temp:
                valid_start = True

            # check if we met either of the start criteria
            if not valid_start:
                continue  # if we did not, continue to next possible start

            # get all the end points after the initial guess
            fwd_ends = candidate_nw_stops[candidate_nw_stops > end_initial]

            try:
                end = fwd_ends[0]
            except IndexError:
                end = avg_temp_delta_5min.size

            # add to list
            nonwear_starts.append(start)
            nonwear_stops.append(end)

            # reset the last end index
            prev_end = end

        # make non-wear indices into arrays, and invert
        wear_starts, wear_stops = invert_indices(
            asarray(nonwear_starts),
            asarray(nonwear_stops),
            0,
            n_ax_under_sd_range_fwd.size,
        )

        # convert to original indices
        wear_starts *= wlen_ds
        wear_stops *= wlen_ds

        # handle case where end is end of array
        if isclose(wear_stops[-1], n_ax_under_sd_range_fwd.size * wlen_ds):
            wear_stops[-1] = time.size - 1

        # create a single wear array, and put it back into the correct
        # units for indexing
        wear = concatenate((wear_starts, wear_stops)).reshape((2, -1)).T

        return {"wear": wear}


class CountWearDetection(BaseProcess):
    r"""
    Detect periods of wear/non-wear from acceleromter data using an implementation
    similar to the ActiGraph counts metric. Consecutive periods of zero activity
    counts are classified as non-wear.

    Parameters
    ----------
    nonwear_window_min : int, optional
        Minutes of zero count to consider nonwear. Default is 90 [2]_.
    epoch_seconds : int, optional
        Number of seconds to accumulate counts for. Default is 60 seconds.
    use_actigraph_package : bool, optional
        Use the internal calculation of activity counts
        (:meth:`skdh.utility.get_activity_counts`), or the Python package published
        by ActiGraph.

    See Also
    --------
    utility.get_activity_counts : activity count calculation

    Notes
    -----
    Note that the internal method for calculating activity counts will give slightly
    different results than the package by ActiGraph, due to handling down-sampling
    differently to handle different devices better.

    References
    ----------
    .. [1] C. E. Matthews et al., “Amount of Time Spent in Sedentary Behaviors in
        the United States, 2003–2004,” American Journal of Epidemiology,
        vol. 167, no. 7, pp. 875–881, Apr. 2008, doi: 10.1093/aje/kwm390.
    .. [2] L. Choi, Z. Liu, C. E. Matthews, and M. S. Buchowski, “Validation of
        Accelerometer Wear and Nonwear Time Classification Algorithm,”
        Medicine & Science in Sports & Exercise, vol. 43, no. 2, pp. 357–364,
        Feb. 2011, doi: 10.1249/MSS.0b013e3181ed61a3.
    """

    def __init__(
        self, nonwear_window_min=90, epoch_seconds=60, use_actigraph_package=False
    ):
        nonwear_window_min = int(nonwear_window_min)
        epoch_seconds = int(epoch_seconds)

        super().__init__(
            nonwear_window_min=nonwear_window_min,
            epoch_seconds=epoch_seconds,
            use_actigraph_package=use_actigraph_package,
        )

        self.nonwear_window_min = nonwear_window_min
        self.epoch_seconds = epoch_seconds
        self.use_ag_package = use_actigraph_package

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, time=None, accel=None, *, fs=None, **kwargs):
        """
        Detect periods of non-wear.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps (in seconds) since 1970-01-01.
        accel : numpy.ndarray
            (N, 3) array of measured acceleration values in units of g.
        fs : float, optional
            Sampling frequency, in Hz. If not provided, will be computed from
            `time`.

        Returns
        -------
        results : dict
            Dictionary of inputs, plus the key `wear` which is an array (N, 2)
            indicating the start and stop indices of wear.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            fs=fs,
            **kwargs,
        )

        # don't start at 0 due to timestamp weirdness with some devices
        fs = 1 / mean(diff(time[1000:5000])) if fs is None else fs

        if self.use_ag_package:
            try:
                from agcounts.extract import get_counts
            except ImportError:
                raise ImportError(
                    "Optional dependency `agcounts` not found. Install using `pip install agcounts`."
                )
            axis_counts = get_counts(
                accel, freq=int(fs), epoch=self.epoch_seconds, fast=True, verbose=False
            )
        else:
            # compute the activity counts
            axis_counts = get_activity_counts(
                fs, time, accel, epoch_seconds=self.epoch_seconds
            )

        # compute single counts vector
        counts = norm(axis_counts, axis=1)

        # get values for specified non-wear window time
        epoch_min = self.epoch_seconds / 60
        wlen = int(self.nonwear_window_min / epoch_min)
        wlen_2 = int(2 / epoch_min)  # TODO make this a parameter?
        wlen_30 = int(30 / epoch_min)  # TODO make this a parameter?

        # 1  : counts > 0
        # 0  : counts == 0
        # -1 : valid nonwear interrupt
        nonwear_counts = (counts > 0).astype(int)
        lengths, starts, values = rle(nonwear_counts)

        # get all instances of less than 2 min windows of interrupts
        idx_lt2 = nonzero((lengths[1:-1] <= wlen_2) & (values[1:-1] == 1))[0] + 1
        # get interrupts with +-30min counts == 0
        mask = (lengths[idx_lt2 - 1] >= wlen_30) & (lengths[idx_lt2 + 1] >= wlen_30)
        idx_lt2 = idx_lt2[mask]

        for s, l in zip(starts[idx_lt2], lengths[idx_lt2]):
            nonwear_counts[s : s + l] = -1

        # get run length encoding again, with modified values for interrupts
        lengths, starts, values = rle(nonwear_counts > 0)

        # get nonwear starts and stops
        mask = (lengths > wlen) & (values == 0)
        nonwear_starts = starts[mask]
        nonwear_stops = nonwear_starts + lengths[mask]

        # invert nonwear to wear
        wear_starts, wear_stops = invert_indices(
            nonwear_starts, nonwear_stops, 0, nonwear_counts.size
        )

        # convert back to original indices
        wear_starts *= int(self.epoch_seconds * fs)
        wear_stops *= int(self.epoch_seconds * fs)

        # handle case where end is end of array
        try:
            if isclose(
                wear_stops[-1], nonwear_counts.size * int(self.epoch_seconds * fs)
            ):
                wear_stops[-1] = time.size - 1
        except IndexError:
            warn("No wear periods detected.")
            pass

        # create a single wear array, and put it back into the correct
        # units for indexing
        wear = concatenate((wear_starts, wear_stops)).reshape((2, -1)).T

        return {"wear": wear}


class CtaWearDetection(BaseProcess):
    r"""
    Detect periods of wear/non-wear from accelerometer and temperature data. CTA
    stands for Combined Temperature and Acceleration.

    Parameters
    ----------
    temp_threshold : float, optional
        Temperature threshold for determining wear or non-wear. From [1]_, 26 deg
        C is the default. NOTE that this was picked for GeneActiv devices,
        and likely is different for other devices, based on where the temperature
        sensor is located. This threshold may also depend on location and
        population.
    sd_crit : float, optional
        Acceleration standard deviation threshold for determining non-wear.
        Default is 0.003 [1]_, which was observed for GeneActiv devices during
        motionless bench tests, and will likely depend on the brand of accelerometer
        being used. 0.013 has also been used [2]_ for this criteria.
    window_length : int, optional
        Length of the windows on which to detect wear and non wear, in minutes. Default is 1 minute.

    References
    ----------
    .. [1] S.-M. Zhou et al., “Classification of accelerometer wear and non-wear events
        in seconds for monitoring free-living physical activity,” BMJ Open,
        vol. 5, no. 5, pp. e007447–e007447, May 2015, doi: 10.1136/bmjopen-2014-007447.
    .. [2] I. C. da Silva et al., “Physical activity levels in three Brazilian birth
        cohorts as assessed with raw triaxial wrist accelerometry,” International
        Journal of Epidemiology, vol. 43, no. 6, pp. 1959–1968, Dec. 2014, doi: 10.1093/ije/dyu203.

    Notes
    -----
    In the original paper _[1] a window skip of 1 second is used to gain sub-minute
    resolution of wear time. However, in this implementation this is dropped,
    as minute-level resolution is already going to be more than enough resolution
    into wear times.
    """

    def __init__(
        self, temp_threshold=26.0, sd_crit=0.003, window_length=1, window_skip=1
    ):
        window_length = int(window_length)
        window_skip = int(window_skip)

        super().__init__(
            temp_thresh=temp_threshold,
            sd_crit=sd_crit,
            window_length=window_length,
            window_skip=window_skip,
        )

        self.temp_thresh = temp_threshold
        self.sd_crit = sd_crit
        self.wlen = window_length
        self.skip = window_skip

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, time=None, accel=None, temperature=None, *, fs=None, **kwargs):
        """
        Detect periods of non-wear.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps (in seconds) since 1970-01-01.
        accel : numpy.ndarray
            (N, 3) array of measured acceleration values in units of g.
        temperature : numpy.ndarray
            (N,) array of measured temperature values during recording in deg C.
        fs : float, optional
            Sampling frequency, in Hz. If not provided, will be computed from
            `time`.

        Returns
        -------
        results : dict
            Dictionary of inputs, plus the key `wear` which is an array-like (N, 2)
            indicating the start and stop indices of wear.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            temperature=temperature,
            fs=fs,
            **kwargs,
        )
        if temperature is None:
            raise ValueError("Temperature is required for this wear algorithm.")

        # dont start at 0 due to timestamp weirdness with some devices
        fs = 1 / mean(diff(time[1000:5000])) if fs is None else fs
        n_wlen = int(self.wlen * 60 * fs)  # samples in `window_length` minutes

        # The original paper uses a skip of 1s to gain sub-minute resolution of
        # non-wear times, however dropping this detail as it really wont make
        # that much of a difference to have that level of resolution

        # compute accel SD for 1 minute non-overlapping windows
        accel_sd = moving_sd(accel, n_wlen, n_wlen, axis=0, return_previous=False)
        # compute moving mean of temperature
        temp_mean = moving_mean(temperature, n_wlen, n_wlen)

        # 3 cases: 1 -> wear, 0 -> non-wear, -1 -> increasing/decreasing rules
        wear = full(temp_mean.size, -1, dtype="int")
        wear[temp_mean >= self.temp_thresh] = 1  # wear if temp is above threshold

        # non-wear - temperature threshold and at least 2 axes have less than sd_crit StDev.
        mask = (temp_mean < self.temp_thresh) & (
            sum(accel_sd < self.sd_crit, axis=1) >= 2
        )
        wear[mask] = 0

        # cases using increasing/decreasing temperature
        # get actual indices for the else case - start @ 1 and add 1 so that
        # we dont try to get index 0 and then compare to index -1
        idx = nonzero(wear[1:] == -1)[0] + 1

        # ELSE case 1 - WEAR: T_t > T_{t-ws}
        # ELSE case 2 - NONWEAR: T_t < T_{t-ws}
        wear[idx] = temp_mean[idx] > temp_mean[idx - 1]

        # ELSE case 3 - unchanged: T_t == T_{t-ws}
        idx3 = idx[temp_mean[idx] == temp_mean[idx - 1]]
        wear[idx3] = wear[idx3 - 1]  # get previous wear status

        # wear is a series of boolean for wear/nonwear, with time deltas of 1minute
        # get the starts and stops of wear time
        lengths, starts, values = rle(wear)

        # convert back to original sampling
        starts *= n_wlen

        wear_starts = starts[values == 1]
        wear_stops = starts[values == 0]

        # check ends
        if not wear[0]:
            wear_stops = wear_stops[1:]
        if wear[-1]:
            wear_stops = append(wear_stops, time.size - 1)

        wear_idx = concatenate((wear_starts, wear_stops)).reshape((2, -1)).T

        return {"wear": wear_idx}


class AccelThresholdWearDetection(BaseProcess):
    r"""
    Detect periods of wear/non-wear from accelerometer data only, based on thresholds of
    the accelerometer standard deviation and range.

    Parameters
    ----------
    sd_crit : float, optional
        Acceleration standard deviation threshold for determining non-wear.
        Default is 0.013 [3]_, which was observed for GeneActiv devices during
        motionless bench tests, and will likely depend on the brand of accelerometer
        being used.
    range_crit : float, optional
        Acceleration window range threshold for determining non-wear. Default is
        0.067, which was found for several GeneActiv accelerometers in a bench
        test as the 75th percentile of the ranges over 60 minute windows.
    apply_setup_criteria : bool, optional
        Apply criteria to the beginning of the recording to account for device setup.
        Default is True.
    shipping_criteria : {bool, int, list}, optional
        Apply shipping criteria to the ends of the trial. Options are False (default,
        no criteria applied), True (criteria applied to the first and last 24 hours),
        an integer (criteria applied to the first and last `shipping_criteria` hours),
        or a length 2 list of integers (criteria applied to the first
        `shipping_criteria[0]` hours and the last `shipping_criteria[1]` hours).
    shipping_temperature : bool, optional
        Apply the `shipping_criteria` rules to `temperature_factor`. For example,
        setting to true would mean with `temperature_factor=2` that during the first
        and last 24 hours (or specified times) the temperature could solely determine
        non-wear. Default is False.
    window_length : int, optional
        Number of minutes in a window used to determine non-wear. Default is 60 minutes.
    window_skip : int, optional
        Number of minutes to skip between windows. Default is 15 minutes, which would
        result in window overlaps of 45 minutes with the default 60-minute
        `window_length`.

    References
    ----------
    .. [1] V. T. van Hees et al., “Separating Movement and Gravity Components in
        an Acceleration Signal and Implications for the Assessment of Human Daily
        Physical Activity,” PLOS ONE, vol. 8, no. 4, p. e61691, Apr. 2013,
        doi: 10.1371/journal.pone.0061691.
    .. [2] V. T. van Hees et al., “Estimation of Daily Energy Expenditure in Pregnant
        and Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer,”
        PLOS ONE, vol. 6, no. 7, p. e22922, Jul. 2011, doi: 10.1371/journal.pone.0022922.
    .. [3] I. C. da Silva et al., “Physical activity levels in three Brazilian birth
        cohorts as assessed with raw triaxial wrist accelerometry,” International
        Journal of Epidemiology, vol. 43, no. 6, pp. 1959–1968, Dec. 2014, doi: 10.1093/ije/dyu203.



    Notes
    -----
    Non-wear is computed by the following:

    .. math::

        NW_{acc} = \sum_{i=\{x,y,z\}}[(a_{(sd, i)} < S) \& (a_{(range, i)} < R)]\\
        NW = NW_{acc} >= 2

    where :math:`a_{sd}` is the acceleration standard deviation of a window,
    :math:`a_{range}` is the range of acceleration of a window, :math:`S` is the
    `sd_crit`, :math:`R` is `range_crit`.

    `apply_setup_criteria` is the rule that if the data starts with a period of non-wear
    of less than 3 hours followed by a non-wear period of any length, then that
    first block of wear is changed to non-wear.

    `shipping_criteria` is an additional rule that may help in cases where the device
    is being shipped either to or from the participant (or both). Wear periods at the
    start of the recording are filtered by those less than 3 hours that are followed
    by 1 hour of non-wear are re-classified as non-wear. Wear periods at the end
    of the recording that are less than 3 hours that are preceded by 1 hour of non-wear
    are re-classified as non-wear.
    """

    def __init__(
        self,
        sd_crit=0.013,
        range_crit=0.067,
        apply_setup_criteria=True,
        shipping_criteria=False,
        shipping_temperature=False,
        window_length=60,
        window_skip=15,
    ):
        window_length = int(window_length)
        window_skip = int(window_skip)
        if isinstance(shipping_criteria, (list, tuple)):
            shipping_criteria = [int(shipping_criteria[i]) for i in range(2)]
        elif isinstance(shipping_criteria, bool):
            if shipping_criteria:
                shipping_criteria = [24, 24]
            else:
                shipping_criteria = [0, 0]
        elif isinstance(shipping_criteria, int):
            shipping_criteria = [shipping_criteria, shipping_criteria]

        super().__init__(
            sd_crit=sd_crit,
            range_crit=range_crit,
            apply_setup_criteria=apply_setup_criteria,
            shipping_criteria=shipping_criteria,
            shipping_temperature=shipping_temperature,
            window_length=window_length,
            window_skip=window_skip,
        )

        self.sd_crit = sd_crit
        self.range_crit = range_crit
        self.apply_setup_crit = apply_setup_criteria
        self.ship_crit = shipping_criteria
        self.ship_temp = shipping_temperature
        self.wlen = window_length
        self.wskip = window_skip

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, *, time, accel, fs=None, **kwargs):
        """
        predict(*, time, accel, fs=None)

        Detect the periods of non-wear

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps (in seconds) since 1970-01-01.
        accel : numpy.ndarray
            (N, 3) array of measured acceleration values in units of g.
        fs : float, optional
            Sampling frequency, in Hz. If not provided, will be computed from
            `time`.

        Returns
        -------
        results : dict
            Dictionary of inputs, plus the key `wear` which is an array-like (N, 2)
            indicating the start and stop indices of wear.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            fs=fs,
            **kwargs,
        )
        # dont start at zero due to timestamp weirdness with some devices
        fs = 1 / mean(diff(time[1000:5000]))
        n_wlen = int(self.wlen * 60 * fs)  # samples in wlen minutes
        n_wskip = int(self.wskip * 60 * fs)  # samples in wskip minutes

        # note that while this block starts at 0, the method uses centered blocks, which
        # means that the first block actually corresponds to a block starting
        # 22.5 minutes into the recording
        acc_rsd = moving_sd(accel, n_wlen, n_wskip, axis=0, return_previous=False)

        # get the accelerometer range in each 60min window
        acc_w_range = moving_max(accel, n_wlen, n_wskip, axis=0) - moving_min(
            accel, n_wlen, n_wskip, axis=0
        )

        nonwear = (
            sum((acc_rsd < self.sd_crit) & (acc_w_range < self.range_crit), axis=1) >= 2
        )

        # flip to wear starts/stops now
        wear_starts, wear_stops = self._modify_wear_times(
            nonwear, self.wskip, self.apply_setup_crit, self.ship_crit
        )

        wear = concatenate((wear_starts, wear_stops)).reshape((2, -1)).T * n_wskip

        return {"wear": wear}

    @staticmethod
    def _modify_wear_times(nonwear, wskip, apply_setup_rule, shipping_crit):
        """
        Modify the wear times based on a set of rules.

        Parameters
        ----------
        nonwear : numpy.ndarray
            Boolean array of nonwear in blocks.
        wskip : int
            Minutes skipped between start of each block.
        apply_setup_rule : bool
            Apply the setup filtering
        shipping_crit : list
            Two element list of number of hours to apply shipping criteria.

        Returns
        -------
        w_starts : numpy.ndarray
            Indices of blocks of wear time starts.
        w_stops : numpy.ndarray
            Indicies of blocks of wear time ends.
        """
        nph = int(60 / wskip)  # number of blocks per hour
        # get the changes in nonwear status
        ch = nonzero(diff(nonwear.astype(int_)))[0] + 1
        ch = insert(
            ch, [0, ch.size], [0, nonwear.size]
        )  # make sure ends are accounted for
        start_with_wear = not nonwear[0]  # does data start with wear period
        end_with_wear = not nonwear[-1]  # does data end with wear period

        # always want to start and end with nonwear, as these blocks wont change
        if start_with_wear:
            ch = insert(
                ch, 0, 0
            )  # extra 0 length nonwear period -> always start with nonwear
        if end_with_wear:
            ch = append(
                ch, nonwear.size
            )  # extra 0 length wear period ->  always end with nonwear

        # pattern is now always [NW][W][NW][W]...[W][NW][W][NW]
        for i in range(3):
            nw_times = (ch[1:None:2] - ch[0:None:2]) / nph  # N
            w_times = (ch[2:-1:2] - ch[1:-1:2]) / nph  # N - 1

            # percentage based rules
            pct = w_times / (nw_times[0:-1] + nw_times[1:None])
            thresh6 = nonzero((w_times <= 6) & (w_times > 3))[0]
            thresh3 = nonzero(w_times <= 3)[0]

            pct_thresh6 = thresh6[pct[thresh6] < 0.3]
            pct_thresh3 = thresh3[pct[thresh3] < 0.8]

            """
            shipping rules
            NOTE: shipping at the start is applied the opposite of shipping at the end, 
            requiring a 1+ hour nonwear period following wear periods less than 3 hours
            """
            ship_start = nonzero(
                (w_times <= 3) & (ch[2:-1:2] <= (shipping_crit[0] * nph))
            )[0]
            ship_end = nonzero(
                (w_times <= 3) & (ch[1:-1:2] >= ch[-1] - (shipping_crit[1] * nph))
            )[0]

            ship_start = ship_start[nw_times[ship_start + 1] >= 1]
            ship_end = ship_end[nw_times[ship_end] >= 1]

            switch = concatenate(
                (
                    pct_thresh6 * 2 + 1,  # start index
                    pct_thresh6 * 2 + 2,  # end index
                    pct_thresh3 * 2 + 1,  # start index
                    pct_thresh3 * 2 + 2,  # end index
                    ship_start * 2 + 1,
                    ship_start * 2 + 2,
                    ship_end * 2 + 1,
                    ship_end * 2 + 2,
                )
            )

            ch = delete(ch, switch)

        w_starts = ch[1:-1:2]
        w_stops = ch[2:-1:2]

        if (
            apply_setup_rule
            and w_starts.size > 0
            and (w_starts[0] == 0)
            and (w_stops[0] <= (3 * nph))
        ):
            w_starts = w_starts[1:]
            w_stops = w_stops[1:]

        return w_starts, w_stops
