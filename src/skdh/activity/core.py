"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from sys import gettrace  # to check if debugging
from datetime import datetime, timedelta
from warnings import warn
from pathlib import Path

from numpy import (
    array,
    mean,
    diff,
    sum,
    zeros,
    abs,
    argmin,
    ceil,
    nan,
    around,
    full,
    arange,
)
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from skdh.base import BaseProcess
from skdh.utility.internal import get_day_index_intersection
from skdh.activity.cutpoints import get_level_thresholds, get_metric
from skdh.activity import endpoints as ept
from skdh.activity.endpoints import ActivityEndpoint
from skdh.activity.utility import handle_cutpoints


def _update_date_results(
    results, time, day_n, day_start_idx, day_stop_idx, day_start_hour
):
    # add 15 seconds to make sure any rounding effects for the hour don't adversely
    # effect the result of the comparison
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
    Classify accelerometer data into different activity levels as a proxy for assessing
    physical activity energy expenditure (PAEE). Levels are sedentary, light, moderate,
    and vigorous. If provided, sleep time will always be excluded from the activity
    level classification.

    Parameters
    ----------
    short_wlen : int, optional
        Short window length in seconds, used for the initial computation acceleration
        metrics. Default is 5 seconds. Must be a factor of 60 seconds.
    max_accel_lens : iterable, optional
        Windows to compute the maximum mean acceleration metric over, in minutes.
        Default is (6, 15, 60).
    bout_lens : iterable, optional
        Activity bout lengths. Default is (1, 5, 10).
    bout_criteria : float, optional
        Value between 0 and 1 for how much of a bout must be above the specified
        threshold. Default is 0.8
    bout_metric : {1, 2, 3, 4, 5}, optional
        How a bout of MVPA is computed. Default is 4. See notes for descriptions
        of each method.
    closed_bout : bool, optional
        If True then count breaks in a bout towards the bout duration. If False
        then only count time spent above the threshold towards the bout duration.
        Only used if `bout_metric=1`. Default is False.
    min_wear_time : int, optional
        Minimum wear time in hours for a day to be analyzed. Default is 10 hours.
    cutpoints : {str, dict, list}, optional
        Cutpoints to use for sedentary/light/moderate/vigorous activity classification.
        Default is "migueles_wrist_adult" [1]_. For a list of all available metrics
        use `skdh.activity.get_available_cutpoints()`. Custom cutpoints can be provided
        in a dictionary (see :ref:`Using Custom Cutpoints`).
    day_window : array-like
        Two (2) element array-like of the base and period of the window to use for
        determining days. Default is (0, 24), which will look for days starting at
        midnight and lasting 24 hours. None removes any day-based windowing.

    Notes
    -----
    While the `bout_metric` methods all should yield fairly similar results, there
    are subtle differences in how the results are computed:

    1. MVPA bout definition from [2]_ and [3]_. Here the algorithm looks for `bout_len`
        minute windows in which more than `bout_criteria` percent of the epochs are
        above the MVPA threshold (above the "light" activity threshold) and then
        counts the entire window as mvpa. The motivation for this definition was
        as follows: A person who spends 10 minutes in MVPA with a 2 minute break
        in the middle is equally active as a person who spends 8 minutes in MVPA
        without taking a break. Therefore, both should be counted equal.
    2. Look for groups of epochs with a value above the MVPA threshold that span
        a time window of at least `bout_len` minutes in which more than `bout_criteria`
        percent of the epochs are above the threshold. Motivation: not counting breaks
        towards MVPA may simplify interpretation and still counts the two persons
        in the above example as each others equal.
    3. Use a sliding window across the data to test `bout_criteria` per window and
        do not allow for breaks larger than 1 minute, and with fraction of time larger
        than the `bout_criteria` threshold.
    4. Same as 3, but also requires the first and last epoch to meet the threshold
        criteria.
    5. Same as 4, but now looks for breaks larger than a minute such that 1 minute
        breaks are allowed, and the fraction of time that meets the threshold should
        be equal or greater than the `bout_criteria` threshold.

    References
    ----------
    .. [1] J. H. Migueles et al., “Comparability of accelerometer signal aggregation
        metrics across placements and dominant wrist cut points for the assessment
        of physical activity in adults,” Scientific Reports, vol. 9, no. 1,
        Art. no. 1, Dec. 2019, doi: 10.1038/s41598-019-54267-y.
    .. [2] I. C. da Silva et al., “Physical activity levels in three Brazilian birth
        cohorts as assessed with raw triaxial wrist accelerometry,” International
        Journal of Epidemiology, vol. 43, no. 6, pp. 1959–1968, Dec. 2014,
        doi: 10.1093/ije/dyu203.
    .. [3] S. Sabia et al., “Association between questionnaire- and
        accelerometer-assessed physical activity: the role of sociodemographic factors,”
        Am J Epidemiol, vol. 179, no. 6, pp. 781–790, Mar. 2014, doi: 10.1093/aje/kwt330.
    """

    act_levels = ["MVPA", "sed", "light", "mod", "vig"]

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
        # make sure that the short_wlen is a factor of 60, and if not send it to
        # nearest factor
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
        cutpoints_ = handle_cutpoints(cutpoints)

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

        # setup wake endpoints
        self.wake_endpoints = [
            ept.IntensityGradient(state="wake"),
            ept.MaxAcceleration(self.max_acc_lens, state="wake"),
        ]
        self.wake_endpoints += [
            ept.TotalIntensityTime(lvl, self.wlen, self.cutpoints, state="wake")
            for lvl in self.act_levels
        ]
        self.wake_endpoints += [
            ept.BoutIntensityTime(
                lvl,
                self.blens,
                self.boutcrit,
                self.boutmetric,
                self.closedbout,
                self.cutpoints,
                state="wake",
            )
            for lvl in self.act_levels
        ]
        self.wake_endpoints += [
            ept.FragmentationEndpoints("sed", cutpoints=cutpoints),
            ept.FragmentationEndpoints("SLPA", cutpoints=cutpoints),
            ept.FragmentationEndpoints("MVPA", cutpoints=cutpoints),
        ]

        self.sleep_endpoints = [
            ept.TotalIntensityTime(lvl, self.wlen, self.cutpoints, state="sleep")
            for lvl in self.act_levels
        ]

    def add(self, endpoint):
        """
        Add an endpoint to the list to be calculated.

        Parameters
        ----------
        endpoint : list, skdh.activity.ActivityEndpoint
            The initialized endpoint, or list of endpoints.
        """
        if isinstance(endpoint, (list, tuple)):
            if all([isinstance(i, ActivityEndpoint) for i in endpoint]):
                for ept in endpoint:
                    if ept.state == "wake":
                        self.wake_endpoints.append(ept)
                    elif ept.state == "sleep":
                        self.sleep_endpoints.append(ept)
                    else:
                        warn(
                            f'Endpoint {ept!r}.state ({ept.state}) not "wake" or '
                            f'"sleep". Skipping'
                        )
        elif isinstance(endpoint, ActivityEndpoint):
            if endpoint.state == "wake":
                self.wake_endpoints.append(endpoint)
            elif endpoint.state == "sleep":
                self.sleep_endpoints.append(endpoint)
            else:
                warn(
                    f'Endpoint {endpoint!r}.state ({endpoint.state}) not "wake" or '
                    f'"sleep". Skipping'
                )
        else:
            warn(f"`endpoint` argument not an ActivityEndpoint or list/tuple. Skipping")

    def _setup_plotting(self, save_name):  # pragma: no cover
        """
        Setup sleep specific plotting

        Parameters
        ----------
        save_name : str
            The file name to save the resulting plot to. Extension will be set to PDF.
            There are formatting options as well for dynamically generated names. See
            Notes.

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
            (N, 3) array of accelerations measured by centrally mounted lumbar device,
            in units of 'g'.
        fs : {None, float}, optional
            Sampling frequency in Hz. If None will be computed from the first 5000
            samples of `time`.
        wear : {None, list}, optional
            List of length-2 lists of wear-time ([start, stop]). Default is None,
            which uses the whole recording as wear time.

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

        # ==============================================================================
        # SETUP / INITIALIZATION
        # ==============================================================================
        if fs is None:
            fs = 1 / mean(diff(time[:5000]))

        nwlen = int(self.wlen * fs)
        nwlen_60 = int(60 * fs)  # minute long windows
        epm = int(60 / self.wlen)  # epochs per minute

        # check if sleep data is provided
        sleep = kwargs.get("sleep", None)
        slp_msg = (
            f"[{self!s}] No sleep information found. Only computing full day metrics."
        )
        sleep_starts, sleep_stops = self._check_if_idx_none(sleep, slp_msg, None, None)

        # ==============================================================================
        # SETUP RESULTS KEYS/ENDPOINTS
        # ==============================================================================
        n_ = self.day_idx[0].size
        res = {
            "Date": full(n_, "", dtype="U11"),
            "Weekday": full(n_, "", dtype="U11"),
            "Day N": full(n_, -1, dtype="int"),
            "N hours": full(n_, nan, dtype="float"),
            "N wear hours": full(n_, nan, dtype="float"),
            "N wear wake hours": full(n_, nan, dtype="float"),
        }

        for endpt in self.wake_endpoints + self.sleep_endpoints:
            if isinstance(endpt.name, (list, tuple)):
                for name in endpt.name:
                    res[name] = full(self.day_idx[0].size, nan, dtype="float")
            else:
                res[endpt.name] = full(self.day_idx[0].size, nan, dtype="float")

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

                res["N wear wake hours"][iday] = around(
                    sum(dwear_stops - dwear_starts) / fs / 3600, 1
                )
            else:
                sleep_wear_starts = sleep_wear_stops = None

            # compute waking hours activity endpoints
            self._compute_awake_activity_endpoints(
                res, accel, fs, iday, dwear_starts, dwear_stops, nwlen, nwlen_60, epm
            )
            # compute sleeping hours activity endpoints
            self._compute_sleep_activity_endpoints(
                res,
                accel,
                fs,
                iday,
                sleep_wear_starts,
                sleep_wear_stops,
                nwlen,
                nwlen_60,
                epm,
            )

        # finalize plots
        self._finalize_plots()

        kwargs.update({self._time: time, self._acc: accel, "fs": fs, "wear": wear})

        return (kwargs, res) if self._in_pipeline else res

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
        for endpt in self.wake_endpoints + self.sleep_endpoints:
            if isinstance(endpt.name, (list, tuple)):
                for name in endpt.name:
                    results[name][day_n] = 0.0
            else:
                results[endpt.name][day_n] = 0.0

    def _compute_awake_activity_endpoints(
        self, results, accel, fs, day_n, starts, stops, n_wlen, n_wlen_60, epm
    ):
        # initialize values from nan to 0.0. Do this here because days with less than
        # minimum hours should have nan values
        self._initialize_awake_values(results, day_n)

        for start, stop in zip(starts, stops):
            # compute the desired acceleration metric
            metric_fn = get_metric(self.cutpoints["metric"])
            acc_metric = metric_fn(
                accel[start:stop], n_wlen, fs, **self.cutpoints["kwargs"]
            )
            acc_metric_60 = metric_fn(
                accel[start:stop], n_wlen_60, fs, **self.cutpoints["kwargs"]
            )

            for endpoint in self.wake_endpoints:
                endpoint.predict(
                    results, day_n, acc_metric, acc_metric_60, self.wlen, epm
                )

        # make sure that any endpoints that were caching values between runs are reset
        for endpoint in self.wake_endpoints:
            endpoint.reset_cached()

    def _compute_sleep_activity_endpoints(
        self, results, accel, fs, day_n, starts, stops, n_wlen, n_wlen_60, epm
    ):
        if starts is None or stops is None:
            return  # don't initialize/compute any values if there is no sleep data

        # initialize values from nan to 0.0. Do this here because days with less than
        # minimum hours should have nan values
        for endpt in self.sleep_endpoints:
            if isinstance(endpt.name, (list, tuple)):
                for name in endpt.name:
                    results[name][day_n] = 0.0
            else:
                results[endpt.name][day_n] = 0.0

        for start, stop in zip(starts, stops):
            metric_fn = get_metric(self.cutpoints["metric"])
            try:
                acc_metric = metric_fn(
                    accel[start:stop], n_wlen, fs, **self.cutpoints["kwargs"]
                )
                acc_metric_60 = metric_fn(
                    accel[start:stop], n_wlen_60, fs, **self.cutpoints["kwargs"]
                )
            except ValueError:  # if not enough points, just skip, value is already set
                continue

            for endpoint in self.sleep_endpoints:
                endpoint.predict(
                    results, day_n, acc_metric, acc_metric_60, self.wlen, epm
                )

        # make sure that any endpoints that were caching values between runs are reset
        for endpoint in self.wake_endpoints:
            endpoint.reset_cached()

    def _plot_day_accel(self, day_n, fs, accel, date_str, start_dt):  # pragma: no cover
        if self.f is None:
            return

        f, ax = plt.subplots(
            nrows=4,
            figsize=(12, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1, 0.5]},
        )
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
                axis="both",
                which="both",
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
        if self.day_key[0] == 0:
            if 0 < 24 - start_hr < 1.5 / 3600:  # within 1.5 seconds
                start_hr -= 24
        x = self._t60 + start_hr
        n60 = int(fs * 60)

        ax[0].plot(
            x[: int(ceil(accel.shape[0] / n60))], accel[: int(1446 * n60) : n60], lw=0.5
        )

        hx = mlines.Line2D([], [], color="C0", label="X", lw=0.5)
        hy = mlines.Line2D([], [], color="C1", label="Y", lw=0.5)
        hz = mlines.Line2D([], [], color="C2", label="Z", lw=0.5)

        ax[0].legend(handles=[hx, hy, hz], bbox_to_anchor=(0, 0.5), loc="center right")

        # compute the metric over 1 minute intervals
        metric_fn = get_metric(self.cutpoints["metric"])
        acc_metric = metric_fn(accel, n60, **self.cutpoints["kwargs"])

        # add to second sub-axis
        ax[1].plot(
            x[: acc_metric.size], acc_metric[:1446], label=self.cutpoints["metric"]
        )
        ax[1].legend(bbox_to_anchor=(0, 0.5), loc="center right")

        # add thresholds to plot
        # do this in reverse order so legend top down is same order as lines
        for thresh in ["moderate", "light", "sedentary"]:
            ax[1].plot(self.day_key, [self.cutpoints[thresh]] * 2, "k--", lw=0.5)

        # labeling the thresholds
        ax[1].text(0, self.cutpoints["moderate"] + 0.025, "vigorous \u2191", color="k")
        ax[1].text(0, self.cutpoints["moderate"] - 0.05, "moderate", color="k")
        ax[1].text(0, self.cutpoints["light"] - 0.05, "light", color="k")
        ax[1].text(0, self.cutpoints["sedentary"] - 0.05, "sedentary", color="k")

        # acceleration level plotting
        acc_level = zeros(acc_metric.size, dtype="int")
        acc_level_text = full(acc_level.size, "", dtype="<U10")
        for i, lvl in enumerate(["sedentary", "light", "moderate", "vigorous"]):
            lthresh, uthresh = get_level_thresholds(lvl, self.cutpoints)

            acc_level[(acc_metric >= lthresh) & (acc_metric < uthresh)] = i
            acc_level_text[(acc_metric >= lthresh) & (acc_metric < uthresh)] = lvl

        ax[2].plot(
            x[: acc_level.size], acc_level, color="k", lw=0.5, label="Accel. Level"
        )
        ax[2].legend(bbox_to_anchor=(0, 0.5), loc="center right")

        ax[-1].set_xlim([self.day_key[0], sum(self.day_key)])
        ax[-1].set_xticks([i for i in range(self.day_key[0], sum(self.day_key) + 1, 3)])
        ax[-1].set_xticklabels([f"{int(i % 24)}:00" for i in ax[-1].get_xticks()])
        ax[-1].set_xlabel("Hour of Day")

    def _plot_day_wear(
        self, fs, day_wear_starts, day_wear_stops, start_dt, day_start
    ):  # pragma: no cover
        if self.f is None:
            return
        start_hr = start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600
        if self.day_key[0] == 0:
            if 0 < 24 - start_hr < 1.5 / 3600:  # within 1.5 seconds
                start_hr -= 24

        wear = []
        for s, e in zip(day_wear_starts - day_start, day_wear_stops - day_start):
            # convert to hours
            sh = s / (fs * 3600) + start_hr
            eh = e / (fs * 3600) + start_hr

            wear.extend([sh, eh, None])  # add None so gaps dont get connected

        self.ax[-1][-1].plot(wear, [2] * len(wear), label="Wear", lw=3)
        self.ax[-1][-1].set_ylim([0.75, 2.25])
        self.ax[-1][-1].legend(bbox_to_anchor=(0, 0.5), loc="center right")

    def _plot_day_sleep(
        self, fs, sleep_starts, sleep_stops, day_start, day_stop, start_dt
    ):  # pragma: no cover
        if self.f is None or sleep_starts is None or sleep_stops is None:
            return

        start_hr = start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600
        if self.day_key[0] == 0:
            if 0 < 24 - start_hr < 1.5 / 3600:  # within 1.5 seconds
                start_hr -= 24
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
        self.ax[-1][-1].legend(bbox_to_anchor=(0, 0.5), loc="center right")

    def _finalize_plots(self):  # pragma: no cover
        if self.f is None:
            return

        date = datetime.today().strftime("%Y%m%d")
        form_fname = self.plot_fname.format(
            date=date, name=self._name, file=self._file_name
        )

        pp = PdfPages(Path(form_fname).with_suffix(".pdf"))

        for f in self.f:
            f.tight_layout()
            f.subplots_adjust(hspace=0)
            pp.savefig(f)

        pp.close()
