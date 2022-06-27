"""
Sleep and major rest period detection

Yiorgos Christakis, Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from sys import gettrace
from collections.abc import Iterable
from warnings import warn
from datetime import datetime, timedelta
from pathlib import Path
from datetime import date as dt_date

from numpy import mean, diff, array, nan, sum, arange, full, int_
from numpy.ma import masked_where
from pandas import DataFrame, date_range
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from skdh.base import BaseProcess  # import the base process class
from skdh.utility.internal import get_day_index_intersection, apply_downsample, rle
from skdh.sleep.tso import get_total_sleep_opportunity
from skdh.sleep.utility import compute_activity_index
from skdh.sleep.sleep_classification import compute_sleep_predictions
from skdh.sleep import endpoints


def _get_date(epoch_ts, day_start_hour):
    """
    Compute the actual day start. Deals with the start days where the day may not start at the day
    windowing time.

    Parameters
    ----------
    epoch_ts : float
        Epoch timestamp in seconds.
    day_start_hour : int
        The hour of the start of a day window

    Returns
    -------
    start_datetime : datetime.datetime
        Datetime of the start of the data.
    day_str : str
        Formatted YYYY-MM-DD string of the start of the day for that window.

    Notes
    -----
    This function works such that if the day window starts at 12:00, and the recording starts at
    10:00, the day returned will be the day *before*, as this would correspond to when the window
    would have started provided the data. This matches the dates schema for the rest of the data.
    """
    # add 15 seconds to make sure any rounding effects for the hour dont adversely effect the
    # result of the comparison
    start_dt = datetime.utcfromtimestamp(epoch_ts + 15)

    if start_dt.hour < day_start_hour:
        day_str = (start_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        day_str = start_dt.strftime("%Y-%m-%d")

    # make sure to remove the 15s from the returned start datetime
    return start_dt - timedelta(seconds=15), day_str


class Sleep(BaseProcess):
    """
    Process raw accelerometer data from the wrist to determine various sleep metrics and endpoints.

    Parameters
    ----------
    start_buffer : int, optional
        Number of seconds to ignore at the beginning of a recording. Default is 0 seconds.
    stop_buffer : int, optional
        Number of seconds to ignore at the end of a recording. Default is 0 seconds.
    internal_wear_temp_thresh : float, optional
        Internal wear calculation temperature threshold in celsius. Internal wear detection is
        performed if no wear is provided, and temperature values exist. Default is 25.0 C. Can
        be disabled by setting to 0.0
    internal_wear_move_thresh : float, optional
        Internal wear calculation movement threshold in g. Internal wear detection is performed if
        no wear is provided, and temperature values are provided. Default is 0.001 g. Can be
        disabled by setting to 0.0
    min_rest_block : int, optional
        Number of minutes required to consider a rest period valid. Default is 30 minutes.
    max_activity_break : int, optional
        Number of minutes of activity allowed to interrupt the major rest period. Default is 30
        minutes.
    tso_min_thresh : float, optional
        Minimum allowed z-angle threshold for determining major rest period. Default is 0.1.
    tso_max_thresh : float, optional
        Maximum allowed z-angle threshold for determining major rest period. Default is 1.0.
    tso_perc : int
        The percentile to use when calculating the TSO threshold from daily data.
        Default is 10.
    tso_factor : float
        The factor to multiply the percentile value by co get the TSO threshold.
        Default is 15.0.
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
    downsample_aa_filter : bool, optional
        Apply an anti-aliasing filter before downsampling. Default is True.
        Uses the same IIR filter as :py:func:`scipy.signal.decimate`.
    day_window : array-like, optional
        Two (2) element array-like of the base and period of the window to use for determining
        days. Default is (12, 24), which will look for days starting at 12 noon and lasting 24
        hours. This should only be changed if the data coming in is from someone who sleeps
        during the day, in which case (0, 24) makes the most sense.
    add_active_time : float, optional
        Add active time to the accelerometer signal start and end when detecting the
        total sleep opportunity. This can occasionally be useful if less than 24 hrs of
        data are collected, as sleep-period skewed data can effect the sleep window
        cutoff, effecting the end results. Suggested is not adding more than 1.5
        hours [5]. Default is 0.0 for no added data.
    save_per_minute_results : bool, optional
        Save minute-by-minute predictions of rest for each day. Default is False.

    Notes
    -----
    Sleep window detection is based off of methods in [1]_, [2]_.

    The detection of sleep and wake states uses a heuristic model based
    on the algorithm described in [3]_.

    The activity index feature is based on the index described in [4]_.

    References
    ----------
    .. [1] van Hees V, Fang Z, Langford J, Assah F, Mohammad Mirkes A, da Silva I, Trenell M,
        White T, Wareham N, Brage S (2014). 'Autocalibration of accelerometer data or free-living
        physical activity assessment using local gravity and temperature: an evaluation on four
        continents.' Journal of Applied Physiology, 117(7), 738-744.
        doi: 10.1152/japplphysiol.00421.2014,
        https://www.physiology.org/doi/10.1152/japplphysiol.00421.2014
    .. [2] van Hees V, Sabia S, Anderson K, Denton S, Oliver J, Catt M, Abell J, Kivimaki M,
        Trenell M, Singh-Maoux A (2015). 'A Novel, Open Access Method to Assess Sleep Duration
        Using a Wrist-Worn Accelerometer.' PloS One, 10(11). doi: 10.1371/journal.pone.0142533,
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142533.
    .. [3] Cole, R.J., Kripke, D.F., Gruen, W.'., Mullaney, D.J., & Gillin, J.C. (1992). Automatic
        sleep/wake identification from wrist activity. Sleep, 15 5, 461-9.
    .. [4] Bai J, Di C, Xiao L, Evenson KR, LaCroix AZ, Crainiceanu CM, et al. (2016) An Activity
        Index for Raw Accelerometry Data and Its Comparison with Other Activity Metrics. PLoS ONE
        11(8): e0160644. https://doi.org/10.1371/journal.pone.0160644
    .. [5] V. T. van Hees et al., “Estimating sleep parameters using an accelerometer
        without sleep diary,” Scientific Reports, vol. 8, no. 1, Art. no. 1, Aug. 2018,
        doi: 10.1038/s41598-018-31266-z.

    """

    _params = [
        # normal metrics
        endpoints.TotalSleepTime,
        endpoints.PercentTimeAsleep,
        endpoints.NumberWakeBouts,
        endpoints.SleepOnsetLatency,
        endpoints.WakeAfterSleepOnset,
        # fragmentation metrics
        endpoints.AverageSleepDuration,
        endpoints.AverageWakeDuration,
        endpoints.SleepWakeTransitionProbability,
        endpoints.WakeSleepTransitionProbability,
        endpoints.SleepGiniIndex,
        endpoints.WakeGiniIndex,
        endpoints.SleepAverageHazard,
        endpoints.WakeAverageHazard,
        endpoints.SleepPowerLawDistribution,
        endpoints.WakePowerLawDistribution,
    ]

    def __init__(
        self,
        start_buffer=0,
        stop_buffer=0,
        internal_wear_temp_thresh=25.0,
        internal_wear_move_thresh=0.001,
        min_rest_block=30,
        max_activity_break=60,
        tso_min_thresh=0.1,
        tso_max_thresh=1.0,
        tso_perc=10,
        tso_factor=15.0,
        min_rest_period=None,
        nonwear_move_thresh=None,
        min_wear_time=0,
        min_day_hours=6,
        downsample=True,
        downsample_aa_filter=True,
        day_window=(12, 24),
        save_per_minute_results=False,
        add_active_time=0.0,
    ):
        super().__init__(
            start_buffer=start_buffer,
            stop_buffer=stop_buffer,
            internal_wear_temp_thresh=internal_wear_temp_thresh,
            internal_wear_move_thresh=internal_wear_move_thresh,
            min_rest_block=min_rest_block,
            max_activity_break=max_activity_break,
            tso_min_thresh=tso_min_thresh,
            tso_max_thresh=tso_max_thresh,
            tso_perc=tso_perc,
            tso_factor=tso_factor,
            min_rest_period=min_rest_period,
            nonwear_move_thresh=nonwear_move_thresh,
            min_wear_time=min_wear_time,
            min_day_hours=min_day_hours,
            downsample=downsample,
            downsample_aa_filter=downsample_aa_filter,
            day_window=day_window,
            save_per_minute_results=save_per_minute_results,
            add_active_time=add_active_time,
        )

        self.window_size = 60
        self.hp_cut = 0.25
        self.start_buff = start_buffer
        self.stop_buff = stop_buffer
        self.int_w_temp = internal_wear_temp_thresh
        self.int_w_move = internal_wear_move_thresh
        self.min_rest_block = min_rest_block
        self.max_act_break = max_activity_break
        self.tso_min_thresh = tso_min_thresh
        self.tso_max_thresh = tso_max_thresh
        self.tso_perc = tso_perc
        self.tso_factor = tso_factor
        self.min_rest_period = min_rest_period
        self.nw_thresh = nonwear_move_thresh
        self.min_wear_time = min_wear_time
        self.min_day_hrs = min_day_hours
        self.downsample = downsample
        self.aa_filter = downsample_aa_filter
        self.save_pm = save_per_minute_results
        self.add_time = add_active_time

        # for storing sleep auxiliary data
        self.sleep_aux = None

        if day_window is None:
            self.day_key = (-1, -1)
        else:
            self.day_key = tuple(day_window)

        # enable plotting as a public method
        self.setup_plotting = self._setup_plotting
        self.t60 = None  # plotting time

    def _setup_plotting(self, save_file):
        """
        Setup sleep specific plotting

        Parameters
        ----------
        save_file : str
            The file name to save the resulting plot to. Extension will be set to PDF. There
            are formatting options as well for dynamically generated names. See Notes

        Notes
        -----
        Available format variables available:

        - date: todays date expressed in yyyymmdd format.
        - name: process name.
        - file: file name used in the pipeline, or "" if not found.
        """
        if save_file is None:
            return
        # move this inside here so that it doesnt effect everything on load
        if gettrace() is None:  # only set if not debugging
            matplotlib.use(
                "PDF"
            )  # non-interactive, dont want to be displaying plots constantly
        plt.style.use("ggplot")

        self.f = []  # need a plot for each day
        self.ax = []  # correspond to each day
        self.plot_fname = save_file

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
        >>> from skdh.sleep.endpoints import SleepEndpoint
        >>> class NewSleepMetric(SleepEndpoint):
        >>>     def __init__(self):
        >>>         super().__init__("metric name", __name__)  # __name__ remains unchanged
        >>>     def predict(self, **kwargs):
        >>>         pass
        >>>
        >>> sleep = Sleep()
        >>> sleep.add_metrics(NewSleepMetric)
        """
        if isinstance(metrics, Iterable):
            if all(isinstance(i(), endpoints.SleepEndpoint) for i in metrics):
                self._params.extend(metrics)
            else:
                raise ValueError("Not all objects are `SleepMetric`s.")
        else:
            if isinstance(metrics(), endpoints.SleepEndpoint):
                self._params.append(metrics)
            else:
                raise ValueError(f"Metric {metrics!r} is not a `SleepMetric`.")

    def save_results(self, results, file_name):
        """
        Save the results of the processing pipeline to a csv file. Will also
        save per minute rest/sleep predictions if `save_per_minute_results` was
        set to `True`. The file name for per minute results is the same as the
        sleep endpoints file with "_per_minute_predictions_day_<n>" added to the end.

        Parameters
        ----------
        results : dict
            Dictionary of results from the output of predict
        file_name : str
            File name. Can be optionally formatted (see Notes)

        Notes
        -----
        Available format variables available:

        - date: todays date expressed in yyyymmdd format.
        - name: process name.
        - file: file name used in the pipeline, or "" if not found.
        """
        file_name = super().save_results(results, file_name)

        if self.save_pm:
            file_name = Path(file_name)

            for i, start in enumerate(self.sleep_aux["start time"]):
                day_n = self.sleep_aux["day n"][i]
                new_name = file_name.stem + f"_per_minute_predictions_day_{day_n}"
                new_name += file_name.suffix
                rest_file = file_name.with_name(new_name)

                tso = self.sleep_aux["tso indices"][i]

                df = DataFrame(columns=["Time", "Rest", "TSO"])
                df["Rest"] = self.sleep_aux["rest predictions"][i]
                df["Time"] = date_range(start, periods=df.shape[0], freq="1T")
                df["TSO"] = False
                df.loc[tso[0] : tso[1], "TSO"] = True

                df.to_csv(rest_file, index=False)

    def predict(
        self, time=None, accel=None, *, temperature=None, fs=None, wear=None, **kwargs
    ):
        """
        predict(time, accel, *, temperature=None, fs=None, wear=None, day_ends={})

        Generate the sleep boundaries and endpoints for a time series signal.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g'
        temperature : numpy.ndarray, optional
            (N, ) array of temperature values, in celsius. Will be used for internal wear
            calculation if `wear` is not provided.
        fs : float, optional
            Sampling frequency in Hz for the acceleration and temperature values. If None,
            will be inferred from the timestamps
        wear : numpy.ndarray, optional
            (P, 2) array of indices indicating where the device is worn. Optional.
        day_ends : dict
            Dictionary containing (N, 2) arrays of start and stop indices for individual days.
            Must have the key
        """
        super().predict(
            expect_days=True,
            expect_wear=True,
            time=time,
            accel=accel,
            temperature=temperature,
            fs=fs,
            wear=wear,
            **kwargs,
        )

        if fs is None:
            fs = 1 / mean(diff(time[:5000]))

        # get the individual days
        days = kwargs.get(self._days, {}).get(self.day_key, None)
        if days is None:
            raise ValueError(
                f"Day indices for {self.day_key} (base, period) not found."
            )

        # get the wear time from previous steps
        if wear is None:
            warn(
                f"[{self!s}] External wear detection not provided. Assuming 100% wear time."
            )
            wear = array([[0, time.size - 1]])

        # downsample if necessary
        goal_fs = 20.0
        if fs != goal_fs and self.downsample:
            (
                time_ds,
                (accel_ds, temp_ds),
                (day_starts_ds, day_stops_ds, wear_starts_ds, wear_stops_ds),
            ) = apply_downsample(
                goal_fs,
                time,
                data=(accel, temperature),
                indices=(*self.day_idx, *self.wear_idx),
                aa_filter=self.aa_filter,
                fs=fs,
            )

        else:
            goal_fs = fs
            time_ds = time
            accel_ds = accel
            temp_ds = temperature
            day_starts_ds, day_stops_ds = self.day_idx
            wear_starts_ds, wear_stops_ds = self.wear_idx

        # setup the storage for the sleep parameters
        sleep = {
            i: []
            for i in [
                "Day N",
                "Date",
                "TSO Start Timestamp",
                "TSO Start",
                "TSO Duration",
            ]
        }

        # iterate over the parameters, initialize them, and put their names into sleep
        init_params = []
        for param in self._params:
            init_params.append(param())
            sleep[init_params[-1].name] = []

        # sleep aux storage if saving per minute results
        if self.save_pm:
            self.sleep_aux = {
                "start time": [],
                "day n": [],
                "rest predictions": [],
                "tso indices": [],
            }

        # setup storage for sleep indices
        sleep_idx = full((day_starts_ds.size, 2), -1, dtype=int_)

        # iterate over the days
        for iday, (start, stop) in enumerate(zip(day_starts_ds, day_stops_ds)):
            if ((stop - start) / (3600 * goal_fs)) < self.min_day_hrs:
                self.logger.info(
                    f"Day {iday} has less than {self.min_day_hrs} hours. Skipping"
                )
                continue

            # initialize all the sleep values for the day
            for k in sleep:
                sleep[k].append(nan)
            # fill out Day number and date
            sleep["Day N"][-1] = iday + 1

            # get the start timestamp and make sure its in the correct hour due to indexing
            start_datetime, sleep["Date"][-1] = _get_date(
                time_ds[start], self.day_key[0]
            )

            # plotting
            source_f = kwargs.get("file", self.plot_fname)
            self._setup_day_plot(iday + 1, source_f, sleep["Date"][-1], start_datetime)
            self._plot_accel(goal_fs, accel_ds[start:stop])

            # get the starts and stops of wear during the day
            dw_starts, dw_stops = get_day_index_intersection(
                wear_starts_ds, wear_stops_ds, True, start, stop
            )

            if (sum(dw_stops - dw_starts) / (3600 * goal_fs)) < self.min_wear_time:
                self.logger.info(
                    f"Day {iday} has less than {self.min_wear_time} externally calculated wear "
                    f"hours. Skipping"
                )
                continue

            # start time, end time, start index, end index
            tso = get_total_sleep_opportunity(
                goal_fs,
                time_ds[start:stop],
                accel_ds[start:stop],
                temp_ds[start:stop] if temp_ds is not None else None,
                dw_starts,
                dw_stops,
                self.min_rest_block,
                self.max_act_break,
                self.tso_min_thresh,
                self.tso_max_thresh,
                self.tso_perc,
                self.tso_factor,
                self.int_w_temp,
                self.int_w_move,
                self._plot_arm_angle,
                idx_start=start,
                add_active_time=self.add_time,
            )

            # calculate activity index
            act_index = compute_activity_index(goal_fs, accel_ds[start:stop])

            self._plot_activity_index(act_index)

            # move this after activity index calculation so that activity index
            # gets plotted always
            if tso[0] is None:
                self._plot_sleep_wear_predictions(
                    goal_fs, None, None, None, dw_starts - start, dw_stops - start
                )
                continue

            # sleep wake predictions
            predictions = compute_sleep_predictions(act_index, sf=0.243)
            # tso indices are already relative to day start
            tso_start = int(tso[2] / int(60 * goal_fs))  # convert to minute indexing
            tso_stop = int(tso[3] / int(60 * goal_fs))
            pred_during_tso = predictions[tso_start:tso_stop]

            # save the sleep per minute results if desired
            self._store_sleep_aux(
                start_datetime, iday, predictions, tso_start, tso_stop
            )

            # set the sleep start and end values as the TSO (essentially time in bed)
            sleep_idx[iday, 0] = int((tso[2] + start) * fs / goal_fs)
            sleep_idx[iday, 1] = int((tso[3] + start) * fs / goal_fs)

            # plotting
            self._plot_sleep_wear_predictions(
                goal_fs,
                predictions,
                tso_start,
                tso_stop,
                dw_starts - start,
                dw_stops - start,
            )

            # results fill out
            tso_start_dt = datetime.utcfromtimestamp(tso[0])
            sleep["TSO Start Timestamp"][-1] = tso[0]
            sleep["TSO Start"][-1] = tso_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            sleep["TSO Duration"][-1] = (tso[3] - tso[2]) / (goal_fs * 60)  # in minutes

            # run length encoding for sleep metrics
            if tso_start == tso_stop:
                continue
            sw_lengths, sw_starts, sw_vals = rle(pred_during_tso)

            for param in init_params:
                sleep[param.name][-1] = param.predict(
                    sleep_predictions=pred_during_tso,
                    lengths=sw_lengths,
                    starts=sw_starts,
                    values=sw_vals,
                )

            self._tabulate_results(sleep)

        # finalize plotting
        self._finalize_plots()

        kwargs.update(
            {
                self._acc: accel,
                self._time: time,
                "fs": fs,
                "wear": wear,
                "temperature": temperature,
                "sleep": sleep_idx,
            }
        )

        return (kwargs, sleep) if self._in_pipeline else sleep

    def _setup_day_plot(self, iday, source_file, date_str, start_dt):
        if self.f is not None:
            f, ax = plt.subplots(
                nrows=4,
                figsize=(12, 6),
                sharex=True,
                gridspec_kw={"height_ratios": [1, 1, 1, 0.5]},
            )

            f.suptitle(
                f"Sleep Visual Report: {Path(source_file).name}\nDay: {iday}\nDate: {date_str}"
            )

            for x in ax:
                x.grid(False)
                x.spines["left"].set_visible(False)
                x.spines["right"].set_visible(False)
                x.spines["top"].set_visible(False)
                x.spines["bottom"].set_visible(False)

                x.tick_params(
                    axis="both",
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    right=False,
                    left=False,
                )

                x.set_yticks([])
                x.set_xticks([])

            self.f.append(f)
            self.ax.append(ax)

            # setup the timestamps for plotting
            start_hr = start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600
            self.t60 = arange(start_hr, self.day_key[1] + self.day_key[0] + 0.1, 1 / 60)
            # pad the end a little to make sure we have enough samples

    def _plot_accel(self, fs, accel):
        """
        Plot the acceleration.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        accel : numpy.ndarray
        """
        if self.f is not None:
            acc = accel[:: int(fs * 60)]

            self.ax[-1][0].plot(self.t60[: acc.shape[0]], acc, lw=0.5)

            hx = mlines.Line2D([], [], color="C0", label="X", lw=0.5)
            hy = mlines.Line2D([], [], color="C1", label="Y", lw=0.5)
            hz = mlines.Line2D([], [], color="C2", label="Z", lw=0.5)
            self.ax[-1][0].legend(
                handles=[hx, hy, hz], bbox_to_anchor=(0, 0.5), loc="center right"
            )

    def _plot_activity_index(self, index):
        """
        Plot the activity measure

        Parameters
        ----------
        index : numpy.ndarray
        """
        if self.f is not None:
            self.ax[-1][1].plot(
                self.t60[: index.size], index, lw=1, color="C3", label="Activity"
            )

            self.ax[-1][1].legend(bbox_to_anchor=(0, 0.5), loc="center right")

    def _plot_arm_angle(self, arm_angle):
        """
        Plot the arm angle

        Parameters
        ----------
        arm_angle : numpy.ndarray
            Arm angle sampled every 5 seconds
        """
        if self.f is not None:
            aa = arm_angle[::12]  # resample to every minute
            self.ax[-1][2].plot(
                self.t60[: aa.size], aa, color="C4", lw=1, label="Arm Angle"
            )

            self.ax[-1][2].legend(bbox_to_anchor=(0, 0.5), loc="center right")

    def _plot_sleep_wear_predictions(
        self, fs, slp, tso_start_i, tso_end_i, wear_starts, wear_stops
    ):
        """
        Plot the sleep predictions

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        slp : numpy.ndarray, None
            Minute-by-minute sleep predictions for the whole day
        tso_start_i : int, None
            Start index for TSO in minute length epochs
        tso_end_i : int, None
            End index for TSO in minute length epochs
        wear_starts : numpy.ndarray
            Indices for wear starts. Indexed to `fs`.
        wear_stops : numpy.ndarray
            Indices for wear ends. Indexed to `fs`.
        """
        if self.f is not None:
            # wear
            h1 = mlines.Line2D(
                [],
                [],
                color="C0",
                label="Wear Prediction",
                lw=3,
                solid_capstyle="round",
            )
            for s, e in zip(wear_starts, wear_stops):
                # convert to hours
                sh = s / (fs * 3600) + self.t60[0]
                eh = e / (fs * 3600) + self.t60[0]
                self.ax[-1][-1].plot(
                    [sh, eh], [2, 2], color="C0", lw=3, solid_capstyle="round"
                )
            # Sleep predictions
            if slp is not None:
                (h2,) = self.ax[-1][-1].plot(
                    self.t60[: slp.size],
                    masked_where(slp == 1, slp) + 1,
                    solid_capstyle="round",
                    lw=3,
                    color="C1",
                    label="Wake Predictions",
                )
            else:
                h2 = mlines.Line2D(
                    [self.t60[0]],
                    [0],
                    solid_capstyle="round",
                    lw=3,
                    color="C1",
                    label="Wake Predictions",
                )
            # Total sleep opportunity
            if tso_start_i is not None and tso_end_i is not None:
                (h3,) = self.ax[-1][-1].plot(
                    [self.t60[tso_start_i], self.t60[tso_end_i]],
                    [0, 0],
                    solid_capstyle="round",
                    lw=3,
                    color="C2",
                    label="Sleep Opportunity",
                )
            else:
                h3 = mlines.Line2D(
                    [self.t60[0]],
                    [0],
                    solid_capstyle="round",
                    lw=3,
                    color="C2",
                    label="Sleep Opportunity",
                )

            self.ax[-1][-1].set_xlim([self.day_key[0], sum(self.day_key)])
            self.ax[-1][-1].set_ylim([-0.25, 2.25])
            self.ax[-1][-1].set_xticks(
                [i for i in range(self.day_key[0], sum(self.day_key) + 1, 3)]
            )
            self.ax[-1][-1].set_xticklabels(
                [f"{int(i % 24)}:00" for i in self.ax[-1][-1].get_xticks()]
            )
            self.ax[-1][-1].set_xlabel("Hour of Day")

            self.ax[-1][-1].legend(
                handles=[h1, h2, h3], bbox_to_anchor=(0, 0.5), loc="center right"
            )

    def _tabulate_results(self, results):
        """
        Put some of the sleep results into a table on the visualization
        """
        keys = [
            "total sleep time",
            "percent time asleep",
            "number of wake bouts",
            "sleep onset latency",
            "wake after sleep onset",
        ]
        if self.f is not None:
            self.ax[-1][0].table(
                [[results[i][-1] for i in keys]], colLabels=keys, loc="top"
            )

    def _finalize_plots(self):
        """
        Finalize and save the plots for sleep
        """
        if self.f is not None:
            date = dt_date.today().strftime("%Y%m%d")
            form_fname = self.plot_fname.format(
                date=date, name=self._name, file=self._file_name
            )
            pp = PdfPages(Path(form_fname).with_suffix(".pdf"))

            for f in self.f:
                f.tight_layout()
                f.subplots_adjust(hspace=0)
                pp.savefig(f)

            pp.close()

    def _store_sleep_aux(self, start_dt, day_n, predictions, tso_start, tso_stop):
        """
        Store the sleep predictions if desired.
        """
        if self.save_pm:
            self.sleep_aux["start time"].append(start_dt)
            self.sleep_aux["day n"].append(day_n)
            self.sleep_aux["rest predictions"].append(predictions)
            self.sleep_aux["tso indices"].append([tso_start, tso_stop])
