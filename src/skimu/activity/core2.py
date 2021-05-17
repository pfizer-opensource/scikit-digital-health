"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Pfizer DMTI 2021
"""
from datetime import datetime, timedelta
from warnings import warn
from pathlib import Path

from numpy import (
    nonzero,
    array,
    mean,
    diff,
    sum,
    zeros,
    abs,
    argmin,
    argmax,
    maximum,
    int_,
    floor,
    ceil,
    histogram,
    log,
    nan,
    around,
    full,
    nanmax,
    arange,
    max,
)
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skimu.base import _BaseProcess
from skimu.utility import moving_mean
from skimu.utility.internal import get_day_index_intersection
from skimu.activity.cutpoints import _base_cutpoints, get_level_thresholds
from skimu.activity import endpoints

# default from rowlands
IG_LEVELS = array([i for i in range(0, 4001, 25)] + [8000]) / 1000
IG_VALS = (IG_LEVELS[1:] + IG_LEVELS[:-1]) / 2


def _check_if_none(var, lgr, msg_if_none, i1, i2):
    if var is None:
        lgr.info(msg_if_none)
        if i1 is None or i2 is None:
            return None, None
        else:
            start, stop = array([i1]), array([i2])
    else:
        start, stop = var[:, 0], var[:, 1]
    return start, stop


def _update_date_results(
    results, time, fs, day_n, day_start_idx, day_stop_idx, day_wear_starts, day_wear_stops, day_start_hour
):
    # add 15 seconds to make sure any rounding effects for the hour don't adversely effect
    # the result of the comparison
    start_dt = datetime.utcfromtimestamp(time[day_start_idx])

    window_start_dt = start_dt + timedelta(seconds=15)
    if start_dt.hour < day_start_hour:
        window_start_dt -= timedelta(days=1)

    results["Date"][day_n] = window_start_dt.strftime("%Y-%m-%d")
    results["Weekday"][day_n] = window_start_dt.strftime("%A")
    results["Day N"][day_n] = day_n + 1
    results["N hours"][day_n] = around(
        (time[day_stop_idx - 1] - time[day_start_idx]) / 3600, 1
    )
    results["N wear hours"][day_n] = around(sum(day_wear_stops - day_wear_starts) / fs / 3600, 1)

    return start_dt


class ActivityLevelClassification(_BaseProcess):
    """
    Classify accelerometer data into different activity levels as a proxy for assessing physical
    activity energy expenditure (PAEE). Levels are sedentary, light, moderate, and vigorous.
    If provided, sleep time will always be excluded from the activity level classification.

    Parameters
    ----------
    short_wlen : int, optional
        Short window length in seconds, used for the initial computation acceleration metrics.
        Default is 5 seconds. Must be a factor of 60 seconds.
    max_accel_lens : iterable, optional
        Windows to compute the maximum mean acceleration metric over, in minutes. Default is
        (6, 15, 60).
    bout_lens : iterable, optional
        Activity bout lengths. Default is (1, 5, 10).
    bout_criteria : float, optional
        Value between 0 and 1 for how much of a bout must be above the specified threshold. Default
        is 0.8
    bout_metric : {1, 2, 3, 4, 5}, optional
        How a bout of MVPA is computed. Default is 4. See notes for descriptions of each method.
    closed_bout : bool, optional
        If True then count breaks in a bout towards the bout duration. If False then only count
        time spent above the threshold towards the bout duration. Only used if `bout_metric=1`.
        Default is False.
    min_wear_time : int, optional
        Minimum wear time in hours for a day to be analyzed. Default is 10 hours.
    cutpoints : {str, dict, list}, optional
        Cutpoints to use for sedentary/light/moderate/vigorous activity classification. Default
        is "migueles_wrist_adult" [1]_. For a list of all available metrics use
        `skimu.activity.get_available_cutpoints()`. Custom cutpoints can be provided in a
        dictionary (see :ref:`Using Custom Cutpoints`).
    day_window : array-like
        Two (2) element array-like of the base and period of the window to use for determining
        days. Default is (0, 24), which will look for days starting at midnight and lasting 24
        hours. None removes any day-based windowing.

    Notes
    -----
    While the `bout_metric` methods all should yield fairly similar results, there are subtle
    differences in how the results are computed:

    1. MVPA bout definition from [2]_ and [3]_. Here the algorithm looks for `bout_len` minute
       windows in which more than `bout_criteria` percent of the epochs are above the MVPA
       threshold (above the "light" activity threshold) and then counts the entire window as mvpa.
       The motivation for this definition was as follows: A person who spends 10 minutes in MVPA
       with a 2 minute break in the middle is equally active as a person who spends 8 minutes in
       MVPA without taking a break. Therefore, both should be counted equal.
    2. Look for groups of epochs with a value above the MVPA threshold that span a time
       window of at least `bout_len` minutes in which more than `bout_criteria` percent of the
       epochs are above the threshold. Motivation: not counting breaks towards MVPA may simplify
       interpretation and still counts the two persons in the above example as each others equal.
    3. Use a sliding window across the data to test `bout_criteria` per window and do not allow
       for breaks larger than 1 minute, and with fraction of time larger than the `bout_criteria`
       threshold.
    4. Same as 3, but also requires the first and last epoch to meet the threshold criteria.
    5. Same as 4, but now looks for breaks larger than a minute such that 1 minute breaks
       are allowed, and the fraction of time that meets the threshold should be equal
       or greater than the `bout_criteria` threshold.

    References
    ----------
    .. [1] J. H. Migueles et al., “Comparability of accelerometer signal aggregation metrics
        across placements and dominant wrist cut points for the assessment of physical activity in
        adults,” Scientific Reports, vol. 9, no. 1, Art. no. 1, Dec. 2019,
        doi: 10.1038/s41598-019-54267-y.
    .. [2] I. C. da Silva et al., “Physical activity levels in three Brazilian birth cohorts as
        assessed with raw triaxial wrist accelerometry,” International Journal of Epidemiology,
        vol. 43, no. 6, pp. 1959–1968, Dec. 2014, doi: 10.1093/ije/dyu203.
    .. [3] S. Sabia et al., “Association between questionnaire- and accelerometer-assessed
        physical activity: the role of sociodemographic factors,” Am J Epidemiol, vol. 179,
        no. 6, pp. 781–790, Mar. 2014, doi: 10.1093/aje/kwt330.
    """

    act_levels = ["MVPA", "sed", "light", "mod", "vig"]
    _max_acc_str = "Max acc {w}min blocks gs"
    _ig_keys = ["IG", "IG intercept", "IG R-squared"]
    _e_wake_str = "{L} {w}s epoch wake mins"
    _e_sleep_str = "{L} {w}s epoch sleep mins"
    _bout_str = "{L} {w}min bout mins"

    def __init__(
        self,
        short_wlen=5,
        max_accel_lens=(6, 15, 60),
        bout_lens=(1, 5, 10),
        bout_criteria=0.8,
        bout_metric=4,
        closed_bout=False,
        min_wear_time=10,
        cutpoints="migueles_wrist_adult",
        day_window=(0, 24),
    ):
        # make sure that the short_wlen is a factor of 60, and if not send it to nearest factor
        if (60 % short_wlen) != 0:
            tmp = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
            short_wlen = tmp[argmin(abs(array(tmp) - short_wlen))]
            warn(f"`short_wlen` changed to {short_wlen} to be a factor of 60.")
        else:
            short_wlen = int(short_wlen)
        # make sure max accel windows are in whole minutes
        max_accel_lens = [int(i) for i in max_accel_lens]
        # make sure bout lengths are in whole minutes
        bout_lens = [int(i) for i in bout_lens]
        # make sure the minimum wear time is in whole hours
        min_wear_time = int(min_wear_time)
        # get the cutpoints if using provided cutpoints, or return the dictionary
        if isinstance(cutpoints, str):
            cutpoints_ = _base_cutpoints.get(cutpoints, None)
            if cutpoints_ is None:
                warn(
                    f"Specified cutpoints [{cutpoints}] not found. Using `migueles_wrist_adult`."
                )
                cutpoints_ = _base_cutpoints["migueles_wrist_adult"]
        else:
            cutpoints_ = cutpoints

        super().__init__(
            short_wlen=short_wlen,
            max_accel_lens=max_accel_lens,
            bout_lens=bout_lens,
            bout_criteria=bout_criteria,
            bout_metric=bout_metric,
            closed_bout=closed_bout,
            min_wear_time=min_wear_time,
            cutpoints=cutpoints_,
            day_window=day_window,
        )

        # setup endpoints
        self.epts = [
            endpoints.ActivityIntensityGradient(width=25, max1=4000, max2=16000),
            endpoints.ActivityEpochMinutes(day_part="wake")
        ]
        self.epts += [
            endpoints.ActivityBoutMinutes(
                bout_length=i,
                bout_criteria=bout_criteria,
                closed_bout=closed_bout,
                bout_metric=bout_metric,
            ) for i in bout_lens
        ]
        self.epts += [
            endpoints.MaximumAcceleration(block_min=i) for i in max_accel_lens
        ]

        self.epts_sleep = [
            endpoints.ActivityEpochMinutes(day_part="sleep")
        ]

        self.wlen = short_wlen
        self.min_wear = min_wear_time
        self.cutpoints = cutpoints_

        if day_window is None:
            self.day_key = (-1, -1)
        else:
            self.day_key = tuple(day_window)

        # enable plotting as a public method
        self.setup_plotting = self._setup_plotting
        self._update_buttons = []
        self._t60 = None  # for storing plotting x values

    def add_endpoints(self, endpoint, for_wake=True):
        """
        Add an endpoint for the results.

        Parameters
        ----------
        endpoint : {skimu.activity.endpoint.ActivityEndpoint, Iterable}
            Single ActivityEndpoint, or an iterable of endpoints, all initialized.
        for_wake : bool, optional
            If the endpoint is for wake time or sleep time. Default is True (wake time).
        """
        if isinstance(endpoint, endpoints.ActivityEndpoint):
            endpoint = [endpoint]
        else:
            if not all([isinstance(i, endpoints.ActivityEndpoint) for i in endpoint]):
                raise ValueError("All endpoints must be initialized subclasses of ActivityEndpoint")

        if for_wake:
            self.epts.extend(list(endpoint))
        else:
            self.epts_sleep.extend(list(endpoint))

    def _setup_plotting(self, save_name):
        """
        Setup sleep specific plotting

        Parameters
        ----------
        save_name : str
            The file name to save the resulting plot to. Extension will be set to PDF. There
            are formatting options as well for dynamically generated names. See Notes

        Notes
        -----
        Available format variables available:

        - date: todays date expressed in yyyymmdd format.
        - name: process name.
        - file: file name used in the pipeline, or "" if not found.
        """
        self.plot_fname = save_name

        self.f = []
        self._t60 = arange(0, 24.1, 1 / 60)

    def predict(self, time=None, accel=None, *, fs=None, wear=None, **kwargs):
        """
        predict(time, accel, *, fs=None, wear=None)

        Compute the time spent in different activity levels.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of continuous unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by centrally mounted lumbar device, in
            units of 'g'
        fs : {None, float}, optional
            Sampling frequency in Hz. If None will be computed from the first 5000 samples of
            `time`.
        wear : {None, list}, optional
            List of length-2 lists of wear-time ([start, stop]). Default is None, which uses the
            whole recording as wear time.

        Returns
        -------
        activity_res : dict
            Computed activity level metrics.
        """
        super().predict(
            expect_days=True,
            expect_wear=True,
            time=time,
            accel=accel,
            fs=fs,
            wear=wear,
            **kwargs
        )

        # =====================================================================
        # SETUP / INITIALIZATION
        # =====================================================================
        fs = 1 / mean(diff(time[:5000])) if fs is None else fs

        nwlen = int(self.wlen * fs)
        # epochs per minute
        epm = int(60 / self.wlen)

        # check if there are sleep windows
        sleep = kwargs.get("sleep", None)
        msg = f"[{self!s}] No sleep windows found. Only computing full day endpoints."
        sleep_starts, sleep_stops = self._check_if_idx_none(sleep, msg, None, None)

        for iday, (dstart, dstop) in enumerate(zip(*self.day_idx)):
            # get the intersection of wear time and day
            day_wear_starts, day_wear_stops = get_day_index_intersection(
                *self.wear_idx, True, dstart, dstop
            )

            # update the date, number of hours, etc
            start_dt = _update_date_results(
                res,
                time,
                fs,
                iday,
                dstart,
                dstop,
                day_wear_starts,
                day_wear_stops,
                self.day_key[0]
            )

            # plotting. handle here before any breaks for minimal wear hours etc
            self._plot_day_accel(
                iday,
                fs,
                accel[dstart:dstop],
                res["Date"][iday],
                start_dt
            )
            self._plot_day_wear(fs, day_wear_starts, day_wear_stops, start_dt, dstart)
            # plotting sleep if it exists
            self._plot_day_sleep(fs, sleep_starts, sleep_stops, dstart, dstop, start_dt)

            # make sure there are enough hours for the endpoints
            if res["N wear hours"][iday] < self.min_wear:
                continue

            # compute the endpoints
            for ept in self.epts:
                res, keys = ept.compute(day_wear_starts, day_wear_stops)
