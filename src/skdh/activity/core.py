"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Pfizer DMTI 2021
"""
from sys import gettrace  # to check if debugging
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
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from skdh.base import BaseProcess
from skdh.utility import moving_mean
from skdh.utility.internal import get_day_index_intersection
from skdh.activity.cutpoints import _base_cutpoints, get_level_thresholds, get_metric


def _update_date_results(
    results, time, day_n, day_start_idx, day_stop_idx, day_start_hour
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

    return start_dt


class ActivityLevelClassification(BaseProcess):
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
        `skdh.activity.get_available_cutpoints()`. Custom cutpoints can be provided in a
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

        self.wlen = short_wlen
        self.max_acc_lens = max_accel_lens
        self.blens = bout_lens
        self.boutcrit = bout_criteria
        self.boutmetric = bout_metric
        self.closedbout = closed_bout
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
        if save_name is None:
            return

        # move this inside here so that it doesnt effect everything on load
        if gettrace() is None:  # only set if not debugging
            matplotlib.use("PDF")  # non-interactiv, dont want to spam plots
        plt.style.use("ggplot")

        self.f = []  # need a plot for each day
        self.ax = []  # correspond to each day
        self.plot_fname = save_name

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
            **kwargs,
        )

        # ========================================================================================
        # SETUP / INITIALIZATION
        # ========================================================================================
        if fs is None:
            fs = 1 / mean(diff(time[:5000]))

        nwlen = int(self.wlen * fs)
        epm = int(60 / self.wlen)  # epochs per minute

        iglevels = (
            array([i for i in range(0, 4001, 25)] + [8000]) / 1000
        )  # default from rowlands
        igvals = (iglevels[1:] + iglevels[:-1]) / 2

        # check if sleep data is provided
        sleep = kwargs.get("sleep", None)
        slp_msg = (
            f"[{self!s}] No sleep information found. Only computing full day metrics."
        )
        sleep_starts, sleep_stops = self._check_if_idx_none(sleep, slp_msg, None, None)

        # =============================================================================
        # SETUP RESULTS KEYS/ENDPOINTS
        # =============================================================================
        res_keys = [("Date", "", "U11"), ("Weekday", "", "U11"), ("Day N", -1, "int")]
        res_keys += [
            ("N hours", nan, "float"),
            ("N wear hours", nan, "float"),
            ("N wear awake hours", nan, "float"),
        ]

        res_keys += [
            (self._max_acc_str.format(w=i), nan, "float") for i in self.max_acc_lens
        ]
        res_keys += [(i, nan, "float") for i in self._ig_keys]
        res_keys += [
            (self._e_wake_str.format(L=i, w=self.wlen), nan, "float")
            for i in self.act_levels
        ]
        res_keys += [
            (self._e_sleep_str.format(L=i, w=self.wlen), nan, "float")
            for i in self.act_levels
        ]
        res_keys += [
            (self._bout_str.format(L=lvl, w=j), nan, "float")
            for lvl in self.act_levels
            for j in self.blens
        ]

        res = {i: full(len(self.day_idx[0]), j, dtype=k) for i, j, k in res_keys}

        # =============================================================================
        # PROCESSING
        # =============================================================================
        for iday, day_idx in enumerate(zip(*self.day_idx)):
            day_start, day_stop = day_idx
            # update the results dictionary with date strings, # of hours, etc
            start_dt = _update_date_results(
                res, time, iday, day_start, day_stop, self.day_key[0]
            )

            # get the intersection of wear time and day
            dwear_starts, dwear_stops = get_day_index_intersection(
                *self.wear_idx, True, day_start, day_stop  # include wear time
            )

            # PLOTTING. handle here before returning for minimal wear hours, etc
            self._plot_day_accel(
                iday,
                fs,
                accel[day_start:day_stop],
                res["Date"][iday],
                start_dt,
            )
            self._plot_day_wear(fs, dwear_starts, dwear_stops, start_dt, day_start)
            # plotting sleep if it exists
            self._plot_day_sleep(
                fs, sleep_starts, sleep_stops, day_start, day_stop, start_dt
            )

            # save wear time and check if there is less wear time than minimum
            res["N wear hours"][iday] = around(
                sum(dwear_stops - dwear_starts) / fs / 3600, 1
            )
            if res["N wear hours"][iday] < self.min_wear:
                continue  # skip day if less than minimum specified hours of wear time

            # if there is sleep data, add it to the intersection of indices
            if sleep_starts is not None and sleep_stops is not None:
                dwear_starts, dwear_stops = get_day_index_intersection(
                    (self.wear_idx[0], sleep_starts),
                    (self.wear_idx[1], sleep_stops),
                    (True, False),  # include wear time, exclude sleeping time
                    day_start,
                    day_stop,
                )
                sleep_wear_starts, sleep_wear_stops = get_day_index_intersection(
                    (self.wear_idx[0], sleep_starts),
                    (self.wear_idx[1], sleep_stops),
                    (True, True),  # now we want only sleep
                    day_start,
                    day_stop,
                )

                res["N wear awake hours"][iday] = around(
                    sum(dwear_starts - dwear_starts) / fs / 3600, 1
                )
            else:
                sleep_wear_starts = sleep_wear_stops = None

            # compute waking hours activity endpoints
            self._compute_awake_activity_endpoints(
                res,
                accel,
                fs,
                iday,
                dwear_starts,
                dwear_stops,
                nwlen,
                epm,
                iglevels,
                igvals,
            )
            # compute sleeping hours activity endpoints
            self._compute_sleep_activity_endpoints(
                res, accel, fs, iday, sleep_wear_starts, sleep_wear_stops, nwlen, epm
            )

        # finalize plots
        self._finalize_plots()

        kwargs.update({self._time: time, self._acc: accel})

        if self._in_pipeline:
            return kwargs, res
        else:
            return res

    def _initialize_awake_values(self, results, day_n):
        """
        Initialize wake results values to 0.0 so they can be added to.

        Parameters
        ----------
        results : dict
            Dictionary of results values
        day_n : int
            Day index value
        """
        for w in self.max_acc_lens:
            results[self._max_acc_str.format(w=w)][day_n] = 0.0
        for lvl in self.act_levels:
            key = self._e_wake_str.format(L=lvl, w=self.wlen)
            results[key][day_n] = 0.0

            for w in self.blens:
                key = self._bout_str.format(L=lvl, w=w)
                results[key][day_n] = 0.0

    def _compute_awake_activity_endpoints(
        self, results, accel, fs, day_n, starts, stops, n_wlen, epm, ig_levels, ig_vals
    ):
        # allocate histogram for intensity gradient
        hist = zeros(ig_levels.size - 1)

        # initialize values from nan to 0.0. Do this here because days with less than
        # minimum hours should have nan values
        self._initialize_awake_values(results, day_n)

        for start, stop in zip(starts, stops):
            # compute the desired acceleration metric
            metric_fn = get_metric(self.cutpoints["metric"])
            acc_metric = metric_fn(
                accel[start:stop], n_wlen, fs, **self.cutpoints["kwargs"]
            )

            # maximum acceleration over windows
            try:
                for max_acc_w in self.max_acc_lens:
                    n = max_acc_w * epm
                    tmp_max = max(moving_mean(acc_metric, n, n))

                    key = self._max_acc_str.format(w=max_acc_w)
                    results[key][day_n] = nanmax([tmp_max, results[key][day_n]])
            except ValueError:
                # if the window length is too long for this block of data
                pass

            # activity levels
            for lvl in self.act_levels:
                # total sum of epochs
                lthresh, uthresh = get_level_thresholds(lvl, self.cutpoints)
                key = self._e_wake_str.format(L=lvl, w=self.wlen)
                results[key][day_n] += (
                    sum((acc_metric >= lthresh) & (acc_metric < uthresh)) / epm
                )

                # time in bouts of specified input lengths
                for bout_len in self.blens:
                    key = self._bout_str.format(L=lvl, w=bout_len)

                    results[key][day_n] += get_activity_bouts(
                        acc_metric,
                        lthresh,
                        uthresh,
                        self.wlen,
                        bout_len,
                        self.boutcrit,
                        self.closedbout,
                        self.boutmetric,
                    )

            # histogram for intensity gradient. Density = false to return counts
            hist += histogram(acc_metric, bins=ig_levels, density=False)[0]

        # intensity gradient computation per day
        hist /= epm  # epm = 60 / self.wlen -> hist *= (self.wlen / 60)
        ig_res = get_intensity_gradient(ig_vals, hist)

        results["IG"][day_n] = ig_res[0]
        results["IG intercept"][day_n] = ig_res[1]
        results["IG R-squared"][day_n] = ig_res[2]

    def _compute_sleep_activity_endpoints(
        self, results, accel, fs, day_n, starts, stops, n_wlen, epm
    ):
        if starts is None or stops is None:
            return  # don't initialize/compute any values if there is no sleep data

        # initialize values from nan to 0.0. Do this here because days with less than minimum
        # hours should have nan values
        for lvl in self.act_levels:
            key = self._e_sleep_str.format(L=lvl, w=self.wlen)
            results[key][day_n] = 0.0

        for start, stop in zip(starts, stops):
            metric_fn = get_metric(self.cutpoints["metric"])
            try:
                acc_metric = metric_fn(
                    accel[start:stop], n_wlen, fs, **self.cutpoints["kwargs"]
                )
            except ValueError:  # if not enough points, just skip, value is already set
                continue

            for lvl in self.act_levels:
                lthresh, uthresh = get_level_thresholds(lvl, self.cutpoints)
                key = self._e_sleep_str.format(L=lvl, w=self.wlen)
                results[key][day_n] += (
                    sum((acc_metric >= lthresh) & (acc_metric < uthresh)) / epm
                )

    def _plot_day_accel(self, day_n, fs, accel, date_str, start_dt):
        if self.f is None:
            return

        f, ax = plt.subplots(nrows=4, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.5]})
        f.suptitle(
            f"Activity Visual Report: {self._file_name}\nDay: {day_n}\nDate: {date_str}"
        )

        for x in ax:
            x.grid(False)
            x.spines["left"].set_visible(False)
            x.spines["right"].set_visible(False)
            x.spines["top"].set_visible(False)
            x.spines["bottom"].set_visible(False)

            x.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                right=False,
                left=False,
            )

            x.set_yticks([])
            x.set_xticks([])

        self.f.append(f)
        self.ax.append(ax)

        start_hr = start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600
        x = self._t60 + start_hr
        n60 = int(fs * 60)

        ax[0].plot(x[:int(ceil(accel.shape[0] / n60))], accel[::n60], lw=0.5)

        hx = mlines.Line2D([], [], color='C0', label='X', lw=0.5)
        hy = mlines.Line2D([], [], color='C1', label='Y', lw=0.5)
        hz = mlines.Line2D([], [], color='C2', label='Z', lw=0.5)

        ax[0].legend(handles=[hx, hy, hz], bbox_to_anchor=(0, 0.5), loc="center right")

        # compute the metric over 1 minute intervals
        metric_fn = get_metric(self.cutpoints["metric"])
        acc_metric = metric_fn(accel, n60, **self.cutpoints["kwargs"])

        # add to second sub-axis
        ax[1].plot(x[:acc_metric.size], acc_metric, label=self.cutpoints['metric'])
        ax[1].legend(bbox_to_anchor=(0, 0.5), loc='center right')

        # add thresholds to plot
        # do this in reverse order so legend top down is same order as lines
        for thresh in ["moderate", "light", "sedentary"]:
            ax[1].plot(self.day_key, [self.cutpoints[thresh]] * 2, 'k--', lw=0.5)

        # labeling the thresholds
        ax[1].text(0, self.cutpoints['moderate'] + 0.025, "vigorous \u2191", color='k')
        ax[1].text(0, self.cutpoints['moderate'] - 0.05, "moderate", color='k')
        ax[1].text(0, self.cutpoints['light'] - 0.05, 'light', color='k')
        ax[1].text(0, self.cutpoints['sedentary'] - 0.05, 'sedentary', color='k')

        # acceleration level plotting
        acc_level = zeros(acc_metric.size, dtype="int")
        acc_level_text = full(acc_level.size, "", dtype="<U10")
        for i, lvl in enumerate(["sedentary", "light", "moderate", "vigorous"]):
            lthresh, uthresh = get_level_thresholds(lvl, self.cutpoints)

            acc_level[(acc_metric >= lthresh) & (acc_metric < uthresh)] = i
            acc_level_text[(acc_metric >= lthresh) & (acc_metric < uthresh)] = lvl

        ax[2].plot(x[:acc_level.size], acc_level, color='k', lw=0.5, label='Accel. Level')
        ax[2].legend(bbox_to_anchor=(0, 0.5), loc='center right')

        ax[-1].set_xlim([self.day_key[0], sum(self.day_key)])
        ax[-1].set_xticks(
            [i for i in range(self.day_key[0], sum(self.day_key) + 1, 3)]
        )
        ax[-1].set_xticklabels(
            [f"{int(i % 24)}:00" for i in ax[-1].get_xticks()]
        )

    def _plot_day_wear(self, fs, day_wear_starts, day_wear_stops, start_dt, day_start):
        if self.f is None:
            return
        start_hr = start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600

        wear = []
        for s, e in zip(day_wear_starts - day_start, day_wear_stops - day_start):
            # convert to hours
            sh = s / (fs * 3600) + start_hr
            eh = e / (fs * 3600) + start_hr

            wear.extend([sh, eh, None])  # add None so gaps dont get connected

        self.ax[-1][-1].plot(wear, [2] * len(wear), label="Wear", lw=3)
        self.ax[-1][-1].set_ylim([0.75, 2.25])
        self.ax[-1][-1].legend(bbox_to_anchor=(0, 0.5), loc='center right')

    def _plot_day_sleep(
        self, fs, sleep_starts, sleep_stops, day_start, day_stop, start_dt
    ):
        if self.f is None or sleep_starts is None or sleep_stops is None:
            return

        start_hr = start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600
        # get day-sleep intersection
        day_sleep_starts, day_sleep_stops = get_day_index_intersection(
            sleep_starts, sleep_stops, True, day_start, day_stop
        )

        sleep = []
        for s, e in zip(day_sleep_starts - day_start, day_sleep_stops - day_start):
            # convert to hours
            sh = s / (fs * 3600) + start_hr
            eh = e / (fs * 3600) + start_hr

            sleep.extend([sh, eh, None])  # add none so it doesn't get connected

        self.ax[-1][-1].plot(sleep, [1] * len(sleep), label="Sleep Opportunity", lw=3)
        self.ax[-1][-1].legend(bbox_to_anchor=(0, 0.5), loc='center right')

    def _finalize_plots(self):
        if self.f is None:
            return

        date = datetime.today().strftime("%Y%m%d")
        form_fname = self.plot_fname.format(
            date=date, name=self._name, file=self._file_name
        )

        pp = PdfPages(Path(form_fname).with_suffix('.pdf'))

        for f in self.f:
            f.tight_layout()
            f.subplots_adjust(hspace=0)
            pp.savefig(f)

        pp.close()


def get_activity_bouts(
    accm, lower_thresh, upper_thresh, wlen, boutdur, boutcrit, closedbout, boutmetric=1
):
    """
    Get the number of bouts of activity level based on several criteria.

    Parameters
    ----------
    accm : numpy.ndarray
        Acceleration metric.
    lower_thresh : float
        Lower threshold for the activity level.
    upper_thresh : float
        Upper threshold for the activity level.
    wlen : int
        Number of seconds in the base epoch
    boutdur : int
        Number of minutes for a bout
    boutcrit : float
        Fraction of the bout that needs to be above the threshold to qualify as a bout.
    closedbout : bool
        If True then count breaks in a bout towards the bout duration. If False then only count
        time spent above the threshold towards the bout duration.
    boutmetric : {1, 2, 3, 4, 5}, optional
        - 1: MVPA bout definition from Sabia AJE 2014 and da Silva IJE 2014. Here the algorithm
            looks for 10 minute windows in which more than XX percent of the epochs are above mvpa
            threshold and then counts the entire window as mvpa. The motivation for the definition
            1 threshold was: A person who spends 10 minutes in MVPA with a 2 minute break in the
            middle is equally active as a person who spends 8 minutes in MVPA without taking a
            break. Therefore, both should be counted equal and as a 10 minute MVPA bout
        - 2: Code looks for groups of epochs with a value above mvpa threshold that span a time
            window of at least mvpadur minutes in which more than BOUTCRITER percent of the epochs
            are above the threshold. Motivation is: not counting breaks towards MVPA may simplify
            interpretation and still counts the two persons in the example as each others equal
        - 3: Use sliding window across the data to test bout criteria per window and do not allow
            for breaks larger than 1 minute and with fraction of time larger than the BOUTCRITER
            threshold.
        - 4: same as 3 but also requires the first and last epoch to meet the threshold criteria.
        - 5: same as 4, but now looks for breaks larger than a minute such that 1 minute breaks
            are allowed, and the fraction of time that meets the threshold should be equal
            or greater than the BOUTCRITER threshold.

    Returns
    -------
    bout_time : float
        Time in minutes spent in bouts of sustained MVPA.
    """
    nboutdur = int(boutdur * (60 / wlen))

    time_in_bout = 0

    if boutmetric == 1:
        x = ((accm >= lower_thresh) & (accm < upper_thresh)).astype(int_)
        p = nonzero(x)[0]
        i_mvpa = 0
        while i_mvpa < p.size:
            start = p[i_mvpa]
            end = start + nboutdur
            if end < x.size:
                if sum(x[start:end]) > (nboutdur * boutcrit):
                    while (sum(x[start:end]) > ((end - start) * boutcrit)) and (
                        end < x.size
                    ):
                        end += 1
                    select = p[i_mvpa:][p[i_mvpa:] < end]
                    jump = maximum(select.size, 1)
                    if closedbout:
                        time_in_bout += (p[argmax(p < end)] - start) * (wlen / 60)
                    else:
                        time_in_bout += jump * (wlen / 60)  # in minutes
                else:
                    jump = 1
            else:
                jump = 1
            i_mvpa += jump
    elif boutmetric == 2:
        x = ((accm >= lower_thresh) & (accm < upper_thresh)).astype(int_)
        xt = zeros(x.size, dtype=int_)
        p = nonzero(x)[0]

        i_mvpa = 0
        while i_mvpa < p.size:
            start = p[i_mvpa]
            end = start + nboutdur
            if end < x.size:
                if sum(x[start : end + 1]) > (nboutdur * boutcrit):
                    xt[start : end + 1] = 2
                else:
                    x[start] = 0
            else:
                if p.size > 1 and i_mvpa > 2:
                    x[p[i_mvpa]] = x[p[i_mvpa - 1]]
            i_mvpa += 1
        x[xt == 2] = 1
        time_in_bout += sum(x) * (wlen / 60)  # in minutes
    elif boutmetric == 3:
        x = ((accm >= lower_thresh) & (accm < upper_thresh)).astype(int_)
        xt = x * 1  # not a view

        # look for breaks larger than 1 minute
        lookforbreaks = zeros(x.size)
        N = int(60 / wlen)
        lookforbreaks[N // 2 : -N // 2 + 1] = moving_mean(x, N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur
        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = moving_mean(
            xt, nboutdur, 1
        )  # window determination can go back to left justified

        p = nonzero(rm > boutcrit)[0]
        for gi in range(nboutdur):
            ind = p + gi
            xt[ind[(ind > 0) & (ind < xt.size)]] = 2
        x[xt != 2] = 0
        x[xt == 2] = 1
        time_in_bout += sum(x) * (wlen / 60)
    elif boutmetric == 4:
        x = ((accm >= lower_thresh) & (accm < upper_thresh)).astype(int_)
        xt = x * 1  # not a view
        # look for breaks longer than 1 minute
        lookforbreaks = zeros(x.size)
        N = int(60 / wlen)
        i1 = int(floor((N + 1) / 2)) - 1
        i2 = int(ceil(x.size - N / 2))
        lookforbreaks[i1:i2] = moving_mean(x, N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur

        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = moving_mean(xt, nboutdur, 1)

        p = nonzero(rm > boutcrit)[0]
        start = int(floor((nboutdur + 1) / 2)) - 1 - int(round(nboutdur / 2))
        # only consider windows that at least start and end with value that meets criteria
        tri = p + start
        tri = tri[(tri > 0) & (tri < (x.size - nboutdur - 1))]
        p = p[nonzero((x[tri] == 1) & (x[tri + nboutdur - 1] == 1))]
        # now mark all epochs that are covered by the remaining windows
        for gi in range(nboutdur):
            ind = p + gi
            xt[ind[nonzero((ind > 0) & (ind < xt.size))]] = 2
        x[xt != 2] = 0
        x[xt == 2] = 1
        time_in_bout += sum(x) * (wlen / 60)
    elif boutmetric == 5:
        x = ((accm >= lower_thresh) & (accm < upper_thresh)).astype(int_)
        xt = x * 1  # not a view
        # look for breaks longer than 1 minute
        lookforbreaks = zeros(x.size)
        N = int(60 / wlen)
        i1 = int(floor((N + 1) / 2)) - 1
        i2 = int(ceil(x.size - N / 2)) - 1
        lookforbreaks[i1:i2] = moving_mean(x, N + 1, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur

        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = moving_mean(xt, nboutdur, 1)

        p = nonzero(rm >= boutcrit)[0]
        start = int(floor((nboutdur + 1) / 2)) - 1 - int(round(nboutdur / 2))
        # only consider windows that at least start and end with value that meets crit
        tri = p + start
        tri = tri[(tri > 0) & (tri < (x.size - nboutdur - 1))]
        p = p[nonzero((x[tri] == 1) & (x[tri + nboutdur - 1] == 1))]

        for gi in range(nboutdur):
            ind = p + gi
            xt[ind[nonzero((ind > 0) & (ind < xt.size))]] = 2

        x[xt != 2] = 0
        x[xt == 2] = 1
        time_in_bout += sum(x) * (wlen / 60)  # in minutes

    return time_in_bout


def get_intensity_gradient(ig_values, counts):
    """
    Compute the intensity gradient metrics from the bin midpoints and the number of minutes spent
    in that bin.

    First computes the natural log of the intensity levels and the minutes in each intensity level.
    Then creates the best linear fit to these data. The slope, y-intercept, and R-squared value
    are of interest for the intensity gradient analysis.

    Parameters
    ----------
    ig_values : numpy.ndarray
        (N, ) array of intensity level bin midpoints
    counts : numpy.ndarray
        (N, ) array of minutes spent in each intensity bin

    Returns
    -------
    gradient : float
        The slope of the natural log of `ig_values` and `counts`.
    intercept : float
        The intercept of the linear regression fit.
    r_squared : float
        R-squared value for the linear regression fit.
    """
    lx = log(
        ig_values[counts > 0] * 1000
    )  # convert back to mg to match GGIR/existing work
    ly = log(counts[counts > 0])

    slope, intercept, rval, *_ = linregress(lx, ly)

    return slope, intercept, rval ** 2
