"""
Wear detection algorithms

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import (
    mean,
    diff,
    sum,
    insert,
    append,
    nonzero,
    delete,
    concatenate,
    int_,
    full,
)

from skdh.base import BaseProcess
from skdh.utility import moving_mean, moving_sd, get_windowed_view
from skdh.utility.internal import rle


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
            self,
            temp_threshold=26.0,
            sd_crit=0.003,
            window_length=1,
            window_skip=1
    ):
        window_length = int(window_length)
        window_skip = int(window_skip)

        super().__init__(
            temp_thresh=temp_threshold,
            sd_crit=sd_crit,
            window_length=window_length,
            window_skip=window_skip
        )

        self.temp_thresh = temp_threshold
        self.sd_crit = sd_crit
        self.wlen = window_length
        self.skip = window_skip

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
        results : dictionary
            Dictionary of inputs, plus the key `wear` which is an array-like (N, 2)
            indicating the start and stop indices of wear.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            temperature=temperature,
            **kwargs
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
        mask = (temp_mean < self.temp_thresh) & (sum(accel_sd < self.sd_crit, axis=1) >= 2)
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
            wear_stops = append(wear_stops, accel.size)

        wear_idx = concatenate((wear_starts, wear_stops)).reshape((-2, 1)).T

        kwargs.update(
            {
                self._time: time,
                self._acc: accel,
                "temperature": temperature,
                "wear": wear_idx,
            }
        )

        return (kwargs, None) if self._in_pipeline else kwargs


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
            shipping_criteria = [24, 24]
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

    def predict(self, time=None, accel=None, temperature=None, **kwargs):
        """
        Detect the periods of non-wear

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps (in seconds) since 1970-01-01.
        accel : numpy.ndarray
            (N, 3) array of measured acceleration values in units of g.
        temperature : numpy.ndarray
            (N,) array of measured temperature values during recording in deg C.

        Returns
        -------
        results : dictionary
            Dictionary of inputs, plus the key `wear` which is an array-like (N, 2)
            indicating the start and stop indices of wear.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            temperature=temperature,
            **kwargs
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
        acc_w = get_windowed_view(accel, n_wlen, n_wskip)
        acc_w_range = acc_w.max(axis=1) - acc_w.min(axis=1)

        nonwear = (
            sum((acc_rsd < self.sd_crit) & (acc_w_range < self.range_crit), axis=1) >= 2
        )

        # flip to wear starts/stops now
        wear_starts, wear_stops = self._modify_wear_times(
            nonwear, self.wskip, self.apply_setup_crit, self.ship_crit
        )

        wear = concatenate((wear_starts, wear_stops)).reshape((2, -1)).T * n_wskip

        kwargs.update({self._time: time, self._acc: accel, "wear": wear, 'temperature': temperature})
        return (kwargs, None) if self._in_pipeline else kwargs

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
        ch = insert(ch, [0, ch.size], [0, nonwear.size])  # make sure ends are accounted for
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
            ship_start = nonzero((w_times <= 3) & (ch[2:-1:2] <= (shipping_crit[0] * nph)))[
                0
            ]
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

        if apply_setup_rule and (w_starts[0] == 0) and (w_stops[0] <= (3 * nph)):
            w_starts = w_starts[1:]
            w_stops = w_stops[1:]

        return w_starts, w_stops
