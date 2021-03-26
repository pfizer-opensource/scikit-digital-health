"""
Sleep and major rest period detection

Yiorgos Christakis, Lukas Adamowicz
Pfizer DMTI 2019-2021
"""
from collections.abc import Iterable
from warnings import warn
from datetime import datetime

from numpy import mean, diff, array, nan

from skimu.base import _BaseProcess  # import the base process class
from skimu.utility.internal import get_day_wear_intersection, apply_downsample, rle
from skimu.sleep.tso import get_total_sleep_opportunity
from skimu.sleep.utility import compute_activity_index
from skimu.sleep.sleep_classification import compute_sleep_predictions
from skimu.sleep.endpoints import *


class Sleep(_BaseProcess):
    """
    Process raw accelerometer data from the wrist to determine various sleep metrics and endpoints.

    Parameters
    ----------
    start_buffer : int, optional
        Number of seconds to ignore at the beginning of a recording. Default is 0 seconds.
    stop_buffer : int, optional
        Number of seconds to ignore at the end of a recording. Default is 0 seconds.
    min_rest_block : int, optional
        Number of minutes required to consider a rest period valid. Default is 30 minutes.
    max_activity_break : int, optional
        Number of minutes of activity allowed to interrupt the major rest period. Default is 30
        minutes.
    min_angle_thresh : float, optional
        Minimum allowed z-angle threshold for determining major rest period. Default is 0.1.
    max_angle_thresh : float, optional
        Maximum allowed z-angle threshold for determining major rest period. Default is 1.0.
    min_rest_period : float, optional
        Minimum length allowed for major rest period. Default is None
    nonwear_move_thresh : float, optional
        Threshold for movement based non-wear. Default is None.
    min_wear_time : float, optional
        Used with `nonwear_move_thresh`.  Wear time in minutes required for data to be considered
        valid. Default is 0
    min_day_hours : float, optional
        Minimum number of hours required to consider a day useable. Default is 6 hours.
    downsample : bool, optional
        Downsample to 20Hz. Default is True.
    day_window : array-like
        Two (2) element array-like of the base and period of the window to use for determining
        days. Default is (12, 24), which will look for days starting at 12 noon and lasting 24
        hours. This should only be changed if the data coming in is from someone who sleeps
        during the day, in which case (0, 24) makes the most sense.

    Notes
    -----
    Sleep window detection is based off of methods in [1]_, [2]_, [3]_.

    The detection of sleep and wake states uses a heuristic model based
    on the algorithm described in [4]_.

    The activity index feature is based on the index described in [5]_.

    References
    ----------
    .. [1] van Hees V, Fang Z, Zhao J, Heywood J, Mirkes E, Sabia S, Migueles J (2019). GGIR: Raw Accelerometer Data Analysis.
        doi: 10.5281/zenodo.1051064, R package version 1.9-1, https://CRAN.R-project.org/package=GGIR.
    .. [2] van Hees V, Fang Z, Langford J, Assah F, Mohammad Mirkes A, da Silva I, Trenell M, White T, Wareham N,
        Brage S (2014). 'Autocalibration of accelerometer data or free-living physical activity assessment using local gravity and
        temperature: an evaluation on four continents.' Journal of Applied Physiology, 117(7), 738-744.
        doi: 10.1152/japplphysiol.00421.2014, https://www.physiology.org/doi/10.1152/japplphysiol.00421.2014
    .. [3] van Hees V, Sabia S, Anderson K, Denton S, Oliver J, Catt M, Abell J, Kivimaki M, Trenell M, Singh-Maoux A (2015).
        'A Novel, Open Access Method to Assess Sleep Duration Using a Wrist-Worn Accelerometer.' PloS One, 10(11).
        doi: 10.1371/journal.pone.0142533, http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142533.
    .. [4] Cole, R.J., Kripke, D.F., Gruen, W.'., Mullaney, D.J., & Gillin, J.C. (1992). Automatic sleep/wake identification
        from wrist activity. Sleep, 15 5, 461-9.
    .. [5] Bai J, Di C, Xiao L, Evenson KR, LaCroix AZ, Crainiceanu CM, et al. (2016) An Activity Index for Raw Accelerometry
        Data and Its Comparison with Other Activity Metrics. PLoS ONE 11(8): e0160644.
        https://doi.org/10.1371/journal.pone.0160644
    """
    _params = [
        # normal metrics
        TotalSleepTime,
        PercentTimeAsleep,
        NumberWakeBouts,
        SleepOnsetLatency,
        WakeAfterSleepOnset,
        # fragmentation metrics
        AverageSleepDuration,
        AverageWakeDuration,
        SleepWakeTransitionProbability,
        WakeSleepTransitionProbability,
        SleepGiniIndex,
        WakeGiniIndex,
        SleepAverageHazard,
        WakeAverageHazard,
        SleepPowerLawDistribution,
        WakePowerLawDistribution
    ]

    def __init__(
            self,
            start_buffer=0,
            stop_buffer=0,
            min_rest_block=30,
            max_activity_break=60,
            min_angle_thresh=0.1,
            max_angle_thresh=1.0,
            min_rest_period=None,
            nonwear_move_thresh=None,
            min_wear_time=0,
            min_day_hours=6,
            downsample=True,
            day_window=(12, 24)
    ):
        super().__init__(
            start_buffer=start_buffer,
            stop_buffer=stop_buffer,
            # temperature_threshold=temperature_threshold,
            min_rest_block=min_rest_block,
            max_activity_break=max_activity_break,
            min_angle_thresh=min_angle_thresh,
            max_angle_thresh=max_angle_thresh,
            min_rest_period=min_rest_period,
            nonwear_move_thresh=nonwear_move_thresh,
            min_wear_time=min_wear_time,
            min_day_hours=min_day_hours,
            downsample=downsample,
            day_window=day_window
        )

        self.window_size = 60
        self.hp_cut = 0.25
        self.start_buff = start_buffer
        self.stop_buff = stop_buffer
        # self.nw_temp = temperature_threshold
        self.min_rest_block = min_rest_block
        self.max_act_break = max_activity_break
        self.min_angle = min_angle_thresh
        self.max_angle = max_angle_thresh
        self.min_rest_period = min_rest_period
        self.nw_thresh = nonwear_move_thresh
        self.min_wear_time = min_wear_time
        self.min_day_hrs = min_day_hours
        self.downsample = downsample

        if day_window is None:
            self.day_key = "-1, -1"
        else:
            self.day_key = f"{day_window[0]}, {day_window[1]}"

    def add_metrics(self, metrics):
        """
        Add metrics to be computed

        Parameters
        ----------
        metrics : {Iterable, callable}
            Either an iterable of EventMetric or BoutMetric references or an individual
            EventMetric/BoutMetric reference to be added to the list of metrics to be computed

        Examples
        --------
        >>> from skimu.sleep.endpoints import SleepMetric
        >>> class NewSleepMetric(SleepMetric):
        >>>     def __init__(self):
        >>>         super().__init__("metric name", __name__)  # __name__ remains unchanged
        >>>     def predict(self, **kwargs):
        >>>         pass
        >>>
        >>> sleep = Sleep()
        >>> sleep.add_metrics(NewSleepMetric)
        """
        if isinstance(metrics, Iterable):
            if all(isinstance(i(), SleepMetric) for i in metrics):
                self._params.extend(metrics)
            else:
                raise ValueError("Not all objects are `SleepMetric`s.")
        else:
            if isinstance(metrics(), SleepMetric):
                self._params.append(metrics)
            else:
                raise ValueError(f"Metric {metrics!r} is not a `SleepMetric`.")

    def predict(self, time=None, accel=None, *, fs=None, wear=None, **kwargs):
        """
        predict(time, accel, *, temp=None, fs=None, day_ends={})

        Generate the sleep boundaries and endpoints for a time series signal.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g'
        fs : float, optional
            Sampling frequency in Hz for the acceleration and temperature values. If None,
            will be inferred from the timestamps
        wear : numpy.ndarray, optional
            (P, 2) array of indices indicating where the device is worn. Optional.
        day_ends : dict
            Dictionary containing (N, 2) arrays of start and stop indices for individual days.
            Must have the key
        """
        if fs is None:
            fs = mean(diff(time[:5000]))

        # get the individual days
        days = kwargs.get(self._days, {}).get("(12, 24)", None)
        if days is None:
            raise ValueError(f"Day indices for {self.day_key} (base, period) not found.")

        # get the wear time from previous steps
        if wear is None:
            warn(f"[{self!s}] Wear detection not provided. Assuming 100% wear time.")
            wear = array([[0, time.size]])

        # downsample if necessary
        goal_fs = 20.
        if fs != goal_fs and self.downsample:
            time_ds, (accel_ds,), (days_ds, wear_ds) = apply_downsample(
                goal_fs, time, data=(accel,), indices=(days, wear)
            )

        else:
            goal_fs = fs
            time_ds = time
            accel_ds = accel
            days_ds = days
            wear_ds = wear

        # setup the storage for the sleep parameters
        sleep = {
            i: [] for i in [
                "Day N", "Date", "TSO Start Timestamp", "TSO Start", "TSO Duration"
            ]
        }

        # iterate over the parameters, initialize them, and put their names into sleep
        init_params = []
        for param in self._params:
            init_params.append(param())
            sleep[init_params[-1].name] = []

        # iterate over the days
        for iday, day_idx in enumerate(days_ds):
            start, stop = day_idx

            # initialize all the sleep values for the day
            for k in sleep:
                sleep[k].append(nan)
            # fill out Day number and date
            sleep["Day N"][-1] = iday + 1
            sleep["Date"][-1] = datetime.utcfromtimestamp(time_ds[start]).strftime("%Y-%m-%d")

            # get the starts and stops of wear during the day
            dw_starts, dw_stops = get_day_wear_intersection(
                wear_ds[:, 0], wear_ds[:, 1], start, stop)

            # start time, end time, start index, end index
            tso = get_total_sleep_opportunity(
                goal_fs,
                time_ds[start:stop],
                accel_ds[start:stop],
                dw_starts,
                dw_stops,
                self.min_rest_block,
                self.max_act_break,
                self.min_angle,
                self.max_angle,
                idx_start=start
            )

            if tso[0] is None:
                continue

            # calculate activity index
            act_index = compute_activity_index(goal_fs, accel_ds[start:stop])

            # sleep wake predictions
            predictions = compute_sleep_predictions(act_index, sf=0.243)
            # tso indices are already relative to day start
            tso_start = int(tso[2] / int(60 * goal_fs))  # convert to minute indexing
            tso_stop = int(tso[3] / int(60 * goal_fs))
            pred_during_tso = predictions[tso_start:tso_stop]
            # run length encoding for sleep metrics
            sw_lengths, sw_starts, sw_vals = rle(pred_during_tso)

            # results fill out
            tso_start_dt = datetime.utcfromtimestamp(tso[0])
            sleep["TSO Start Timestamp"][-1] = tso[0]
            sleep["TSO Start"][-1] = tso_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            sleep["TSO Duration"][-1] = (tso[3] - tso[2]) / (goal_fs * 60)  # in minutes

            for param in init_params:
                sleep[param.name][-1] = param.predict(
                    sleep_predictions=pred_during_tso,
                    lengths=sw_lengths,
                    starts=sw_starts,
                    values=sw_vals
                )

        kwargs.update({self._acc: accel, self._time: time, "fs": fs, "wear": wear})
        if self._in_pipeline:
            return kwargs, sleep
        else:
            return sleep

