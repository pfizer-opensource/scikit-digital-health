"""
Activity level classification based on accelerometer data

Lukas Adamowicz
Pfizer DMTI 2021
"""
from numpy import nonzero, array, insert, append, mean, diff, sum

from skimu.base import _BaseProcess
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
    min_wear_time : int, optional
        Minimum wear time in hours for a day to be analyzed. Default is 10 hours.
    cutpoints : {str, dict, list}, optional
        Cutpoints to use for sedentary/light/moderate/vigorous activity classification. Default
        is "migueles_wrist_adult". For a list of all available metrics use
        `skimu.activity.get_available_cutpoints()`. Custom cutpoints can be provided in a
        dictionary (see :ref:`Using Custom Cutpoints`).
    """
    def __init__(self, min_wear_time=10, cutpoints="migueles_wrist_adult"):
        min_wear_time = int(min_wear_time)
        cutpoints = _base_cutpoints.get(cutpoints, _base_cutpoints["migueles_wrist_adult"])

        super().__init__(
            min_wear_time=min_wear_time,
            cutpoints=cutpoints
        )

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

            for dwstart, dwstop in zip(day_wear_starts, day_wear_stops):
                hours = (dwstop - dwstart) / fs / 3600
                # compute the metric for the acceleration
                accel_metric = self.cutpoints["metric"](
                    accel[dwstart:dwstop], int(fs), fs,
                    **self.cutpoints["kwargs"]
                )

                # get the starts and stops of different activity levels
                lt_sed = accel_metric < self.cutpoints["sedentary"]
                lt_light = accel_metric < self.cutpoints["light"]
                lt_mod = accel_metric < self.cutpoints["moderate"]

                sed_starts, sed_stops = _get_level_starts_stops(lt_sed)
                light_starts, light_stops = _get_level_starts_stops(~lt_sed & lt_light)
                mod_starts, mod_stops = _get_level_starts_stops(~lt_light & lt_mod)
                vig_starts, vig_stops = _get_level_starts_stops(~lt_mod)



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
