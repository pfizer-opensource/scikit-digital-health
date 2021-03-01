"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Pfizer DMTI 2021
"""
from warnings import warn

from numpy import nonzero, array, insert, append, mean, diff, sum, zeros, abs, argmin, max

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
    min_wear_time : int, optional
        Minimum wear time in hours for a day to be analyzed. Default is 10 hours.
    cutpoints : {str, dict, list}, optional
        Cutpoints to use for sedentary/light/moderate/vigorous activity classification. Default
        is "migueles_wrist_adult". For a list of all available metrics use
        `skimu.activity.get_available_cutpoints()`. Custom cutpoints can be provided in a
        dictionary (see :ref:`Using Custom Cutpoints`).
    """
    def __init__(self, short_wlen=5, bout_len1=1, bout_len2=5, bout_len3=10, min_wear_time=10,
                 cutpoints="migueles_wrist_adult"):
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
            min_wear_time=min_wear_time,
            cutpoints=cutpoints
        )

        self.wlen = short_wlen
        self.blen1 = bout_len1
        self.blen2 = bout_len2
        self.blen3 = bout_len3
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

                # time in bout1 minutes
                nbout = self.blen1 * (60 / self.wlen)




                # get the starts and stops of different activity levels
                lt_sed = accel_metric < self.cutpoints["sedentary"]
                lt_light = accel_metric < self.cutpoints["light"]
                lt_mod = accel_metric < self.cutpoints["moderate"]

                # make sure the indices are into the full accel array
                sed_starts, sed_stops = _get_level_starts_stops(lt_sed) + dwstart
                light_starts, light_stops = _get_level_starts_stops(~lt_sed & lt_light) + dwstart
                mod_starts, mod_stops = _get_level_starts_stops(~lt_light & lt_mod) + dwstart
                vig_starts, vig_stops = _get_level_starts_stops(~lt_mod) + dwstart


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
        time spent above the threshold towards the bout duration
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

    """
    nboutdur = boutdur * (60 / wlen)

    time_in_bout = 0

    if boutmetric == 1:
        x = zeros(accm.size + 1)
        x[1:] = accm > mvpa_thresh  # makes it so that dont have to worry about inserting an index
        starts = nonzero(diff(x.astype("int")) == 1)[0]
        x = x[1:]  # dont need that start now

        for start in starts:
            end = start + nboutdur
            if end < x.size:
                if sum(x[start:end]) > (nboutdur * boutcrit):
                    while sum(x[start:end]) > ((end - start) * boutcrit) and (end < x.size):
                        end += 1
                    time_in_bout +=



    p = nonzero(x)[0]
    if boutmetric == 1:
        xt = x.astype("int")
        boutcount = zeros(x.size)
        jmvpa = 1  # index
        Lx = x.size
        while jmvpa <= p.size:
            endi = p[jmvpa] + boutdur
            if endi <= Lx:  # does bout fall within measurement
                if sum(x[p[jmvpa]:endi]) > (boutdur * boutcrit):
                    while (sum(x[p[jmvpa]:endi]) > ((endi - p[jmvpa]) * boutcrit)) and (endi < Lx):
                        endi += 1
                    select = p[jmvpa:max(nonzero(p < endi)[0])]
                    jump = select.size
                    xt[select] = 2  # remember this was a bout
                    boutcount[p[jmvpa]:p[max(nonzero(p < endi))]] = 1
                else:
                    jump = 1
                    x[p[jmvpa]] = 0
            else:
                jump = 1
                if (p.size > 1) and (jmvpa > 2):
                    x[p[jmvpa]] = x[p[jmvpa - 1]]
            jmvpa = jmvpa + jump
        x[xt == 2] = 1
        if nonzero(xt == 1)[0].size > 0:
            x[xt == 1] = 0
        if closedbout:
            x = boutcount
    elif boutmetric == 2:  # MVPA based on percentage relative to start of bout
        xt = x.astype("int")
        jmvpa = 1
        while jmvpa <= p.size:
            endi = p[jmvpa] + boutdur
            if endi < x.size:
                lengthbout = sum(x[p[jmvpa]:endi])
                if lengthbout > (boutdur * boutcrit):
                    xt[p[jmvpa]:endi] = 2
                else:
                    x[p[jmvpa]] = 0
            else:
                if (p.size > 1) and (jmvpa > 2):
                    x[p[jmvpa]] = x[p[jmvpa - 1]]
            jmvpa += 1
        x[xt == 2] = 1
        boutcount = x
    elif boutmetric == 3:  # simply look at % of moving window that meets criteria
        xt = x.astype("int")
        # look for breaks larger than 1 minute
        lookforbreaks = zeros(x.size)
        N = int(60 / wlen)
        lookforbreaks[N//2:-N//2] = rolling_mean(x.astype("int"), N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks] = -N * boutdur
        # in this way there will not be bout breaks lasting longer than 1 minute
        RM = zeros(xt.size)
        RM[boutdur//2:-boutdur//2] = rolling_mean(xt, boutdur, 1)
        p = nonzero(RM > boutcrit)[0]
        starti = int(round(boutdur / 2))
        for gi in range(boutdur):
            inde = p - starti + gi
            xt[inde[(inde > 0) & (inde < xt.size)]] = 2
        x[xt != 2] = 0
        x[xt == 2] = 1
        boutcount = x
    elif boutmetric == 4:
        xt = x.astype("int")
        # look for breaks larger than 1 minute
        lookforbreaks = zeros(x.size)
        N = int(60 / wlen)
        lookforbreaks[N // 2:-N // 2] = rolling_mean(x.astype("int"), N, 1)
        # insert negative numbers to prevent these minutes from being counted in bouts
        xt[lookforbreaks] = -N * boutdur
        # in this way there will not be bout breaks lasting longer than 1 minute
        RM = zeros(xt.size)
        RM[boutdur // 2:-boutdur // 2] = rolling_mean(xt, boutdur, 1)
        p = nonzero(RM > boutcrit)[0]
        starti = int(round(boutdur / 2))
        # only consider windows that at least start and end with value that meets criteria
        tri = p - starti
        kep = nonzero((tri > 0) & (tri < (x.size - (boutdur - 1))))[0]
        if kep.size > 0:
            tri = tri[kep]
        p = p[nonzero(x[tri] == 1 & (x[tri + (boutdur - 1)] == 1))[0]]
        # mark all epochs that are covered by remaining windows
        for gi in range(boutdur):
            inde = p - starti + gi
            xt[inde[nonzero(inde > 0 & (inde < x.size))]] = 2
        x[xt != 2] = 0
        x[xt == 2] = 0
        boutcount = x
    elif boutmetric == 5:




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
