"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Pfizer DMTI 2021
"""
from warnings import warn

from numpy import nonzero, array, insert, append, mean, diff, sum, zeros, abs, argmin, argmax, \
    maximum, int_, floor, ceil

from skimu.base import _BaseProcess
from skimu.utility import rolling_mean
from skimu.activity.metrics import *

# ==========================================================
# Activity cutpoints
_base_cutpoints = {}

_base_cutpoints["esliger_lwrist_adult"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": True},
    "sedentary": 217 / 80 / 60,  # paper at 80hz, summed for each minute long window
    "light": 644 / 80 / 60,
    "moderate": 1810 / 80 / 60
}

_base_cutpoints["esliger_rwirst_adult"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": True},
    "sedentary": 386 / 80 / 60,  # paper at 80hz, summed for each 1min window
    "light": 439 / 80 / 60,
    "moderate": 2098 / 80 / 60
}

_base_cutpoints["esliger_lumbar_adult"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": True},
    "sedentary": 77 / 80 / 60,  # paper at 80hz, summed for each 1min window
    "light": 219 / 80 / 60,
    "moderate": 2056 / 80 / 60
}

_base_cutpoints["schaefer_ndomwrist_child6-11"] = {
    "metric": metric_bfen,
    "kwargs": {"low_cutoff": 0.2, "high_cutoff": 15, "trim_zero": False},
    "sedentary": 0.190,
    "light": 0.314,
    "moderate": 0.998
}

_base_cutpoints["phillips_rwrist_child8-14"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": True},
    "sedentary": 6 / 80,  # paper at 80hz, summed for each 1s window
    "light": 21 / 80,
    "moderate": 56 / 80
}

_base_cutpoints["phillips_lwrist_child8-14"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": True},
    "sedentary": 7 / 80,
    "light": 19 / 80,
    "moderate": 60 / 80
}

_base_cutpoints["phillips_hip_child8-14"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": True},
    "sedentary": 3 / 80,
    "light": 16 / 80,
    "moderate": 51 / 80
}

_base_cutpoints["vaha-ypya_hip_adult"] = {
    "metric": metric_mad,
    "kwargs": {},
    "light": 0.091,  # originally presented in mg
    "moderate": 0.414
}

_base_cutpoints["hildebrand_hip_adult_actigraph"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0474,
    "light": 0.0691,
    "moderate": 0.2587
}

_base_cutpoints["hildebrand_hip_adult_geneactv"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0469,
    "light": 0.0687,
    "moderate": 0.2668
}

_base_cutpoints["hildebrand_wrist_adult_actigraph"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0448,
    "light": 0.1006,
    "moderate": 0.4288
}

_base_cutpoints["hildebrand_wrist_adult_geneactiv"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0458,
    "light": 0.0932,
    "moderate": 0.4183
}

_base_cutpoints["hildebrand_hip_child7-11_actigraph"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0633,
    "light": 0.1426,
    "moderate": 0.4646
}

_base_cutpoints["hildebrand_hip_child7-11_geneactiv"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0641,
    "light": 0.1528,
    "moderate": 0.5143
}

_base_cutpoints["hildebrand_wrist_child7-11_actigraph"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0356,
    "light": 0.2014,
    "moderate": 0.707
}

_base_cutpoints["hildebrand_wrist_child7-11_geneactiv"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0563,
    "light": 0.1916,
    "moderate": 0.6958
}

_base_cutpoints["migueles_wrist_adult"] = {
    "metric": metric_enmo,
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.050,
    "light": 0.110,
    "moderate": 0.440
}


def get_available_cutpoints():
    """
    Print the available cutpoints for activity level segmentation.
    """
    print(_base_cutpoints.keys())


class MVPActivityClassification(_BaseProcess):
    """
    Classify accelerometer data into different activity levels as a proxy for assessing physical
    activity energy expenditure (PAEE). Levels are sedentary, light, moderate, and vigorous.

    Parameters
    ----------
    short_wlen : int, optional
        Short window length in seconds, used for the initial computation acceleration metrics.
        Default is 5 seconds. Must be a factor of 60 seconds.
    bout_len1 : int, optional
        Activity bout length 1, in minutes. Default is 1 minutes.
    bout_len2 : int, optional
        Activity bout length 2, in minutes. Default is 5 minutes.
    bout_len3 : int, optional
        Activity bout length 3, in minutes. Default is 10 minutes.
    bout_criteria : float, optional
        Value between 0 and 1 for how much of a bout must be above the specified threshold. Default
        is 0.8
    bout_metric : {1, 2, 3, 4, 5}, optional
        How a bout of MVPA is computed. Default is 1. See notes for descriptions of each method.
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
            self, short_wlen=5, bout_len1=1, bout_len2=5, bout_len3=10, bout_criteria=0.8,
            bout_metric=1, closed_bout=False, min_wear_time=10, cutpoints="migueles_wrist_adult"
    ):
        if (60 % short_wlen) != 0:
            tmp = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
            short_wlen = tmp[argmin(abs(array(tmp) - short_wlen))]
            warn(f"`short_wlen` changed to {short_wlen} to be a factor of 60.")
        else:
            short_wlen = int(short_wlen)
        bout_len1 = int(bout_len1)
        bout_len2 = int(bout_len2)
        bout_len3 = int(bout_len3)
        min_wear_time = int(min_wear_time)
        cutpoints = _base_cutpoints.get(cutpoints, _base_cutpoints["migueles_wrist_adult"])

        super().__init__(
            short_wlen=short_wlen,
            bout_len1=bout_len1,
            bout_len2=bout_len2,
            bout_len3=bout_len3,
            bout_criteria=bout_criteria,
            bout_metric=bout_metric,
            closed_bout=closed_bout,
            min_wear_time=min_wear_time,
            cutpoints=cutpoints
        )

        self.wlen = short_wlen
        self.blen1 = bout_len1
        self.blen2 = bout_len2
        self.blen3 = bout_len3
        self.boutcrit = bout_criteria
        self.boutmetric = bout_metric
        self.closedbout = closed_bout
        self.min_wear = min_wear_time
        self.cutpoints = cutpoints

    def predict(self, time=None, accel=None, *, wear=None, **kwargs):
        """
        predict(time, accel, *, wear=None)

        Compute the time spent in different activity levels.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of continuous unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by centrally mounted lumbar device, in
            units of 'g'
        wear : {None, list}, optional
            List of length-2 lists of wear-time ([start, stop]). Default is None, which uses the
            whole recording as wear time.

        Returns
        -------
        activity_res : dict
            Computed activity level metrics.
        """
        super().predict(time=time, accel=accel, wear=wear, **kwargs)

        # longer than it needs to be really, but due to weird timesamps for some devices
        # using this for now
        fs = 1 / mean(diff(time))

        nwlen = int(self.wlen * fs)
        nblen1 = int(self.blen1 * 60 * fs)
        nblen2 = int(self.blen2 * 60 * fs)

        if wear is None:
            self.logger.info(f"[{self!s}] Wear detection not provided. Assuming 100% wear time.")
            wear_starts = array([0])
            wear_stops = array([time.size])
        else:
            tmp = array(wear).astype("int")  # make sure it can broadcast correctly
            wear_starts = tmp[:, 0]
            wear_stops = tmp[:, 1]

        days = kwargs.get(self._days, [[0, time.size - 1]])

        for iday, day_idx in enumerate(days):
            start, stop = day_idx

            # get the intersection of wear time and day
            day_wear_starts, day_wear_stops = get_day_wear_intersection(
                wear_starts, wear_stops, start, stop)

            # less wear time than minimum
            day_wear_hours = sum(day_wear_stops - day_wear_starts) / fs / 3600
            if day_wear_hours < self.min_wear:
                continue  # skip day

            mvpa = zeros(6)  # per epoch, per 1min epoch, per 5min epoch, per bout lengths

            for dwstart, dwstop in zip(day_wear_starts, day_wear_stops):
                hours = (dwstop - dwstart) / fs / 3600
                # compute the metric for the acceleration
                accel_metric = self.cutpoints["metric"](
                    accel[dwstart:dwstop], nwlen, fs,
                    **self.cutpoints["kwargs"]
                )

                # MVPA
                # total time of `wlen` epochs in minutes
                mvpa[0] += sum(accel_metric >= self.cutpoints["light"]) * (self.wlen / 60)
                # total time of 1 minute epochs
                tmp = rolling_mean(accel_metric, int(60 / self.wlen), int(60 / self.wlen))
                mvpa[1] += sum(tmp >= self.cutpoints["light"])  # already in minutes
                # total time in 5 minute epochs
                tmp = rolling_mean(accel_metric, int(300 / self.wlen), int(300 / self.wlen))
                mvpa[2] += sum(tmp >= self.cutpoints["light"]) * 5  # in minutes

                # total time in bout1 minute bouts
                mvpa[3] += get_activity_bouts(
                    accel_metric, self.cutpoints["light"], self.wlen, self.blen1, self.boutcrit,
                    self.closedbout, self.boutmetric
                )
                # total time in bout2 minute bouts
                mvpa[4] += get_activity_bouts(
                    accel_metric, self.cutpoints["light"], self.wlen, self.blen2, self.boutcrit,
                    self.closedbout, self.boutmetric
                )
                # total time in bout3 minute bouts
                mvpa[5] += get_activity_bouts(
                    accel_metric, self.cutpoints["light"], self.wlen, self.blen3, self.boutcrit
                )


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
        lookforbreaks[N // 2:-N // 2 + 1] = rolling_mean(x, N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur
        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = rolling_mean(xt, nboutdur, 1)  # window determination can go back to left justified

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
        lookforbreaks[i1:i2] = rolling_mean(x, N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur

        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = rolling_mean(xt, nboutdur, 1)

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
        lookforbreaks[i1:i2] = rolling_mean(x, N + 1, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks == 0] = -(60 / wlen) * nboutdur

        # in this way there will not be bout breaks lasting longer than 1 minute
        rm = rolling_mean(xt, nboutdur, 1)

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


def get_day_wear_intersection(starts, stops, day_start, day_stop):
    """
    Get the intersection between wear times and day starts/stops.

    Parameters
    ----------
    starts : numpy.ndarray
        Array of integer indices where gait bouts start.
    stops : numpy.ndarray
        Array of integer indices where gait bouts stop.
    day_start : int
        Index of the day start.
    day_stop : int
        Index of the day stop.

    Returns
    -------
    day_wear_starts : numpy.ndarray
        Array of wear start indices for the day.
    day_wear_stops : numpy.ndarray
        Array of wear stop indices for the day
    """
    day_start, day_stop = int(day_start), int(day_stop)
    # get the portion of wear times for the day
    starts_subset = starts[(starts >= day_start) & (starts < day_stop)]
    stops_subset = stops[(stops > day_start) & (stops <= day_stop)]

    if starts_subset.size == 0 and stops_subset.size == 0:
        if stops[nonzero(starts <= day_start)[0][-1]] >= day_stop:
            return array(day_start), array(day_stop)
        else:
            return array([0]), array([0])
    if starts_subset.size == 0 and stops_subset.size == 1:
        starts_subset = array([day_start])
    if starts_subset.size == 1 and stops_subset.size == 0:
        stops_subset = array([day_stop])

    if starts_subset[0] > stops_subset[0]:
        starts_subset = insert(starts_subset, 0, day_start)
    if starts_subset[-1] > stops_subset[-1]:
        stops_subset = append(stops_subset, day_stop)

    assert starts_subset.size == stops_subset.size, "bout starts and stops do not match"

    return starts_subset, stops_subset


def _get_level_starts_stops(mask):
    """
    Get the start and stop indices for a mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean numpy array.

    Returns
    -------
    starts : numpy.ndarray
        Starts of `True` value blocks in `mask`.
    stops : numpy.ndarray
        Stops of `True` value blocks in `mask`.
    """
    delta = diff(mask.astype("int"))

    starts = nonzero(delta == 1)[0] + 1
    stops = nonzero(delta == -1)[0] + 1

    if starts[0] > stops[0]:
        starts = insert(starts, 0, 0)
    if stops[-1] < starts[-1]:
        stops = append(stops, mask.size)

    return starts, stops
