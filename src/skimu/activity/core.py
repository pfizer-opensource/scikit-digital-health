"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Pfizer DMTI 2021
"""
from datetime import datetime
from warnings import warn
from itertools import product as iter_product

from numpy import nonzero, array, insert, append, mean, diff, sum, zeros, abs, argmin, argmax, \
    maximum, int_, floor, ceil, histogram, log, nan, around, argsort, full
from scipy.stats import linregress

from skimu.base import _BaseProcess
from skimu.utility import moving_mean
from skimu.utility.internal import get_day_index_intersection
from skimu.activity.cutpoints import _base_cutpoints


def _check_if_none(var, lgr, msg_if_none, i1, i2):
    if var is None:
        lgr.info(msg_if_none)
        start, stop = array([i1]), array([i2])
    else:
        start, stop = var[:, 0], var[:, 1]
    return start, stop


def _update_date_results(results, timestamps, day_n, day_start_idx, day_stop_idx):
    day_date = datetime.utcfromtimestamp(timestamps[day_start_idx] + 5)
    results["Date"][day_n] = day_date.strftime("%Y-%m-%d")
    results["Weekday"][day_n] = day_date.strftime("%A")
    results["Day N"][day_n] = day_n + 1
    results["N hours"][day_n] = around(
        (timestamps[day_stop_idx - 1] - timestamps[day_start_idx]) / 3600,
        1
    )


class MVPActivityClassification(_BaseProcess):
    """
    Classify accelerometer data into different activity levels as a proxy for assessing physical
    activity energy expenditure (PAEE). Levels are sedentary, light, moderate, and vigorous.

    Parameters
    ----------
    short_wlen : int, optional
        Short window length in seconds, used for the initial computation acceleration metrics.
        Default is 5 seconds. Must be a factor of 60 seconds.
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
    def __init__(
            self, short_wlen=5, bout_lens=(1, 5, 10), bout_criteria=0.8,
            bout_metric=4, closed_bout=False, min_wear_time=10, cutpoints="migueles_wrist_adult",
            day_window=(0, 24)
    ):
        if (60 % short_wlen) != 0:
            tmp = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
            short_wlen = tmp[argmin(abs(array(tmp) - short_wlen))]
            warn(f"`short_wlen` changed to {short_wlen} to be a factor of 60.")
        else:
            short_wlen = int(short_wlen)
        bout_lens = [int(i) for i in bout_lens]
        min_wear_time = int(min_wear_time)
        if isinstance(cutpoints, str):
            cutpoints = _base_cutpoints.get(cutpoints, _base_cutpoints["migueles_wrist_adult"])

        super().__init__(
            short_wlen=short_wlen,
            bout_lens=bout_lens,
            bout_criteria=bout_criteria,
            bout_metric=bout_metric,
            closed_bout=closed_bout,
            min_wear_time=min_wear_time,
            cutpoints=cutpoints
        )

        self.wlen = short_wlen
        self.blens = bout_lens
        self.boutcrit = bout_criteria
        self.boutmetric = bout_metric
        self.closedbout = closed_bout
        self.min_wear = min_wear_time
        self.cutpoints = cutpoints

        if day_window is None:
            self.day_key = (-1, -1)
        else:
            self.day_key = tuple(day_window)

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
        super().predict(time=time, accel=accel, fs=fs, wear=wear, **kwargs)

        # ========================================================================================
        # SETUP / INITIALIZATION
        # ========================================================================================
        if fs is None:
            fs = 1 / mean(diff(time[:5000]))

        nwlen = int(self.wlen * fs)
        epm = int(60 / self.wlen)  # epochs per minute

        iglevels = array([i for i in range(0, 4001, 25)] + [8000]) / 1000  # default from rowlands
        igvals = (iglevels[1:] + iglevels[:-1]) / 2

        wear_none_msg = f"[{self!s}] Wear detection not provided. Assuming 100% wear time."
        wear_starts, wear_stops = _check_if_none(wear, self.logger, wear_none_msg, 0, time.size)

        # check if windows exist for days
        days = kwargs.get(self._days, {}).get(self.day_key, None)
        if days is None:
            warn(
                f"Day indices for {self.day_key} (base, period) not found. No day separation used"
            )
            days = [[0, time.size]]

        # check if sleep data is provided
        sleep = kwargs.get("sleep", None)
        slp_msg = f"[{self!s}] No sleep information found. Only computing full day metrics."
        sleep_starts, sleep_stops = _check_if_none(sleep, self.logger, slp_msg, 0, time.size)

        # ========================================================================================
        # SETUP RESULTS KEYS/ENDPOINTS
        # ========================================================================================
        general_str_keys = ["Date", "Weekday"]
        general_int_keys = ["Day N", "N Hours", "N wear hours", "N wear awake hours"]

        # MM: midnight -> midnight    ExS: Exclude Sleep
        windows = ["MM", "ExS"]
        activity_levels = ["MVPA", "sed", "light", "mod", "vig"]

        lvl_keys = iter_product(windows, activity_levels, self.blens, ["bout"])
        mvpa_keys = iter_product(windows, ["MVPA"], ["5sec", "1min", "5min"], ["epoch"])
        ig_keys = iter_product(windows, ["IG"], ["gradient", "intercept", "R-squared"])

        res = {i: full(len(days), "", dtype="object") for i in general_str_keys}
        res.update({i: full(len(days), -1, dtype="int") for i in general_int_keys})
        res.update({i: full(len(days), nan, dtype="float") for i in mvpa_keys})
        res.update({i: full(len(days), nan, dtype="float") for i in lvl_keys})
        res.update({i: full(len(days), nan, dtype="float") for i in ig_keys})

        # ========================================================================================
        # PROCESSING
        # ========================================================================================
        for iday, day_idx in enumerate(days):
            day_start, day_stop = day_idx

            _update_date_results(res, time, iday, day_start, day_stop)

            # get the intersection of wear time and day
            day_wear_starts, day_wear_stops = get_day_index_intersection(
                wear_starts,
                wear_stops,
                True,  # include wear time
                day_start,
                day_stop
            )
            # get the intersection of wear time, wake time, and day
            day_wear_wake_starts, day_wear_wake_stops = get_day_index_intersection(
                (wear_starts, sleep_starts),
                (wear_stops, sleep_stops),
                (True, False),  # include wear time, exclude sleeping time
                day_start,
                day_stop
            )

            # less wear time than minimum
            res["N wear hours"][iday] = around(sum(day_wear_stops - day_wear_starts) / fs / 3600, 1)
            res["N wear awake hours"][iday] = around(
                sum(day_wear_wake_stops - day_wear_wake_starts) / fs / 3600, 1
            )
            if res["N wear hours"][iday] < self.min_wear:
                continue  # skip day

            # intensity gradient should be done on the whole days worth of data
            hist = zeros(iglevels.size - 1)

            for dwstart, dwstop in zip(day_wear_starts, day_wear_stops):
                # compute the metric for the acceleration
                accel_metric = self.cutpoints["metric"](
                    accel[dwstart:dwstop], nwlen, fs,
                    **self.cutpoints["kwargs"]
                )

                # MVPA
                # total time of `wlen` epochs in minutes
                res[("MM", "MVPA", "5sec", "epoch")][-1] += sum(
                    accel_metric >= self.cutpoints["light"]) / epm
                # total time of 1 minute epochs
                tmp = moving_mean(accel_metric, epm, epm)
                res[("MM", "MVPA", "1min", "epoch")][-1] += sum(tmp >= self.cutpoints["light"])
                # total time in 5 minute epochs
                tmp = moving_mean(accel_metric, 5 * epm, 5 * epm)
                res[("MM", "MVPA", "5min", "epoch")][-1] += sum(tmp >= self.cutpoints["light"]) * 5

                # total MVPA time in <bout_len> minute bouts
                for bout_len in self.blens:
                    res[f"MVPA {bout_len}min Bouts"][-1] += get_activity_bouts(
                        accel_metric, self.cutpoints["light"], self.wlen, bout_len, self.boutcrit,
                        self.closedbout, self.boutmetric
                    )

                # intensity gradient. Density = false to return counts
                hist += histogram(accel_metric, bins=iglevels, density=False)[0]

            # intensity gradient computation per day
            hist *= (self.wlen / 60)
            ig_res = get_intensity_gradient(igvals, hist)
            res[("MM", "IG", "gradient")][iday] = ig_res[0]
            res[("MM", "IG", "intercept")][iday] = ig_res[1]
            res[("MM", "IG", "R-squared")][iday] = ig_res[2]

        kwargs.update({self._time: time, self._acc: accel})

        if self._in_pipeline:
            return kwargs, res
        else:
            return res


def get_activity_bouts(accm, mvpa_thresh, wlen, boutdur, boutcrit, closedbout, boutmetric=1):
    """
    Get the number of bouts of activity level based on several criteria.

    Parameters
    ----------
    accm : numpy.ndarray
        Acceleration metric.
    mvpa_thresh : float
        Threshold for determining moderate/vigorous physical activity.
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
        x = (accm > mvpa_thresh).astype(int_)
        p = nonzero(x)[0]
        i_mvpa = 0
        while i_mvpa < p.size:
            start = p[i_mvpa]
            end = start + nboutdur
            if end < x.size:
                if sum(x[start:end]) > (nboutdur * boutcrit):
                    while (sum(x[start:end]) > ((end - start) * boutcrit)) and (end < x.size):
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
        x = (accm > mvpa_thresh).astype(int_)
        xt = zeros(x.size, dtype=int_)
        p = nonzero(x)[0]

        i_mvpa = 0
        while i_mvpa < p.size:
            start = p[i_mvpa]
            end = start + nboutdur
            if end < x.size:
                if sum(x[start:end + 1]) > (nboutdur * boutcrit):
                    xt[start:end + 1] = 2
                else:
                    x[start] = 0
            else:
                if p.size > 1 and i_mvpa > 2:
                    x[p[i_mvpa]] = x[p[i_mvpa - 1]]
            i_mvpa += 1
        x[xt == 2] = 1
        time_in_bout += sum(x) * (wlen / 60)  # in minutes
    elif boutmetric == 3:
        x = (accm > mvpa_thresh).astype(int_)
        xt = x * 1  # not a view

        # look for breaks larger than 1 minute
        lookforbreaks = zeros(x.size)
        N = int(60 / wlen)
        lookforbreaks[N // 2:-N // 2 + 1] = moving_mean(x, N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur
        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = moving_mean(xt, nboutdur, 1)  # window determination can go back to left justified

        p = nonzero(rm > boutcrit)[0]
        for gi in range(nboutdur):
            ind = p + gi
            xt[ind[(ind > 0) & (ind < xt.size)]] = 2
        x[xt != 2] = 0
        x[xt == 2] = 1
        time_in_bout += sum(x) * (wlen / 60)
    elif boutmetric == 4:
        x = (accm > mvpa_thresh).astype(int_)
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
        x = (accm > mvpa_thresh).astype(int_)
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
    lx = log(ig_values[counts > 0] * 1000)  # convert back to mg to match GGIR/existing work
    ly = log(counts[counts > 0])

    slope, intercept, rval, *_ = linregress(lx, ly)

    return slope, intercept, rval**2


def get_day_sleep_intersection(day_start, day_stop, sleep):
    """
    Get the end of sleep and start of sleep for a day given the day start and stop indices.

    Parameters
    ----------
    day_start : int
        Index of the day start.
    day_stop : int
        Index of the day stop.
    sleep : numpy.ndarray
        Array of sleep start and stop indices. (N, 2) shape, where [:, 0] is the index for sleep
        starts, and [:, 1] is the index for sleep stops

    Returns
    -------
    day_wake_start : int
        Index of when the day starts and subject is awake.
    day_wake_stop : int
        Index of when the day ends and the subject is asleep.
    """
    if sleep is None:
        return None, None

    wake_changes = []
    wake_states = []

    mask = (sleep[:, 1] < day_stop) & (sleep[:, 1] > day_start)
    wake_changes.extend(sleep[mask, 1].to_list())
    wake_states.extend(["start"] * mask.sum())

    mask = (sleep[:, 0] < day_stop) & (sleep[:, 0] > day_start)
    wake_changes.extend(sleep[mask, 0].to_list())
    wake_states.extend(["stop"] * mask.sum())

    if wake_changes == []:
        return

    idx = argsort(wake_changes)

    wake_changes = array(wake_changes)[idx]
    wake_states = array(wake_states)[idx]

    if wake_states[0] == "stop":
        wake_changes = insert(wake_changes, 0, day_start)
        wake_states = insert(wake_states, 0, "start")

    if wake_states[-1] == "start":
        wake_changes = append(wake_changes, day_stop)
        wake_states = append(wake_states, "stop")

    # now pattern should always be [start][stop]...



