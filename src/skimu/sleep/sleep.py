"""
Sleep and major rest period detection

Yiorgos Christakis
Pfizer DMTI 2019-2021
"""
# TODO: build predict function using tso.py, activity_index.py, sleep_classification.py, endpoints.py
from numpy import zeros, arange, interp, float_, mean, diff

from skimu.base import _BaseProcess  # import the base process class
from skimu.sleep.tso import detect_tso


class Sleep(_BaseProcess):
    """
    Process raw accelerometer data from the wrist to determine various sleep metrics and endpoints.

    Parameters
    ----------
    start_buffer : int, optional
        Number of seconds to ignore at the beginning of a recording. Default is 0 seconds.
    stop_buffer : int, optional
        Number of seconds to ignore at the end of a recording. Default is 0 seconds.
    temperature_threshold : float, optional
        Lowest temperature for which a data point is considered valid/wear. Default is 25C.
    min_rest_block : int, optional
        Number of minutes required to consider a rest period valid. Default is 30 minutes.
    allowed_rest_break : int, optional
        Number of minutes allowed to interrupt the major rest period. Default is 30 minutes
    min_rest_threshold : float, optional
        Minimum allowed z-angle threshold for determining major rest period. Default is 0.1.
    max_rest_threshold : float, optional
        Maximum allowed z-angle threshold for determining major rest period. Default is 1.0.
    min_rest_period : float, optional
        Minimum length allowed for major rest period. Default is None
    movement_based_nonwear : float, optional
        Threshold for movement based non-wear. Default is None.
    min_wear_time : float, optional
        Used with `movement_based_nonwear`.  Wear time in minutes required for data to be considered
        valid. Default is 0
    minimum_hours : float, optional
        Minimum number of hours required to consider a day useable. Default is 6 hours.
    downsample : bool, optional
        Downsample to 20Hz. Default is True.

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
    def __init__(
            self, start_buffer=0, stop_buffer=0, temperature_threshold=25, min_rest_block=30,
            allowed_rest_break=30, min_rest_threshold=0.1, max_rest_threshold=1.0,
            min_rest_period=None, movement_based_nonwear=None, min_wear_time=0,
            minimum_hours=6, downsample=True
    ):
        super().__init__(
            start_buffer=start_buffer,
            stop_buffer=stop_buffer,
            temperature_threshold=temperature_threshold,
            min_rest_block=min_rest_block,
            allowed_rest_break=allowed_rest_break,
            min_rest_threshold=min_rest_threshold,
            max_rest_threshold=max_rest_threshold,
            min_rest_period=min_rest_period,
            movement_based_nonwear=movement_based_nonwear,
            min_wear_time=min_wear_time,
            minimum_hours=minimum_hours,
            downsample=downsample
        )

        self.window_size = 60
        self.hp_cut = 0.25
        self.start_buff = start_buffer
        self.stop_buff = stop_buffer
        self.T_min = temperature_threshold
        self.min_rest_block = min_rest_block
        self.allowed_rest_break = allowed_rest_break
        self.min_rest_thresh = min_rest_threshold
        self.max_rest_thresh = max_rest_threshold
        self.min_rest_period = min_rest_period
        self.move_based_nwear = movement_based_nonwear
        self.min_wear_time = min_wear_time
        self.downsample = downsample

    def predict(self, time=None, accel=None, *, temp=None, fs=None, **kwargs):
        """
        predict(time, accel, *, temp=None, fs=None)

        Generate the sleep boundaries and endpoints for a time series signal.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g'
        temp : numpy.ndarray, optional
            (N, ) array of temperature values, units of 'C'
        fs : float, optional
            Sampling frequency in Hz for the acceleration and temperature values. If None,
            will be inferred from the timestamps
        """
        if fs is None:
            fs = mean(diff(time[:5000]))

        # downsample if necessary
        goal_fs = 20.
        if fs != goal_fs and self.downsample:
            # get timestamps
            time_ds = arange(time[0], time[-1], 1 / 20.0)

            # get acceleration
            accel_ds = zeros((time_ds.size, 3), dtype=float_)
            for i in range(3):
                accel_ds[:, i] = interp(time_ds, time, accel[:, i])

            # get temp
            if temp is not None:
                temp_ds = zeros(time_ds.size, dtype=float_)
                temp_ds[:, 0] = interp(time_ds, time, temp)
            else:
                temp_ds = None
        else:
            goal_fs = fs
            time_ds = time
            accel_ds = accel
            temp_ds = temp

        tso = detect_tso(accel_ds, time_ds, goal_fs, temp_ds, self.min_rest_block, )
        # FULL SLEEP PIPELINE
        # compute total sleep opportunity window
        # compute activity index
        # compute sleep predictions
        # compute sleep related endpoints


