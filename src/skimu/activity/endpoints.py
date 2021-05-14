"""
File containing the activity endpoint classes

Lukas Adamowicz
Pfizer DMTI 2021
"""
import logging

from numpy import max, sum, array, zeros, histogram, log
from scipy.stats import linregress

from skimu.utility import moving_mean
from skimu.activity.cutpoints import get_level_thresholds
from skimu.activity.core import get_activity_bouts


class ActivityEndpoint:
    """
    Activity endpoint base class.

    Parameters
    ----------
    res_names : tuple of str
        Results names/keys.
    logname : str
        Name of the logger to get
    """
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self, res_names, logname):
        self.res_names = res_names
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(logname)

    def compute(self, *args, **kwargs):
        pass


class ActivityIntensityGradient(ActivityEndpoint):
    r"""
    Compute the intensity gradient - a measure of the drop-off in increasing
    activity level.

    Notes
    -----
    The intensity gradient calculation is a way of assessing the profile
    of the activity levels. The accelerometer data is binned, and the natural
    log of the counts in each bin are taken, along with the natural log of
    the bin midpoint. These two sets of values are used to generate a linear
    regression, with the intensity gradient being the slope. Also of interest
    are the y intercept and the :math:`r^2` value.

    The bins are computed using the following:

    .. math:: bin_edges = [0, w, 2w, 3w, ...., m_1 - 2w, m_1 - w, m_1, m_2]

    where :math:`w` is the `width`, :math:`m_1` is `max1`, and :math:`m_2`
    is `max_2.

    References
    ----------
    .. [1] A. V. Rowlands, C. L. Edwardson, M. J. Davies, K. Khunti,
        D. M. Harrington, and T. Yates, “Beyond Cut Points: Accelerometer
        Metrics that Capture the Physical Activity Profile,” Medicine &
        Science in Sports & Exercise, vol. 50, no. 6, pp. 1323–1332, Jun. 2018,
        doi: 10.1249/MSS.0000000000001561.
    """
    def __init__(self, **kwargs):
        super().__init__(
            (
                "Intensity Gradient",
                "IG intercept",
                "IG r-sq",
            ),
            __name__,
        )

        self.ig_levels = None
        self.ig_vals = None

        self.set_bin_values()  # to defaults

    def set_bin_values(self, width=25, max1=4000, max2=16000):
        """
        Parameters
        ----------
        width : int
            Bin width for the histogram bins, in milli-g. Default is 25mg (0.025g).
        max1 : int
            Maximum value of normal bin widths, in milli-g. Default is 4000mg (4g).
            This value is inclusive on the end.
        max2 : int
            Maximum second value of the last bin, in milli-g. Default is 16000mg (8g).
        """
        self.ig_levels = array([i for i in range(0, max1 + 1, width)] + [max2])
        self.ig_vals = (self.ig_levels[1:] + self.ig_levels[:-1]) / 2
        # store ig_vals in milli-g to match existing work, but the actual
        # levels to get from accel should be in g
        self.ig_levels /= 1000

    def compute(self, starts, stops, fs, wlen, cutpoints, acc_metric):
        """
        Compute the number of total minutes spent in an activity level.

        Parameters
        ----------
        starts : numpy.ndarray
            Start indices of valid bouts of data.
        stops : numpy.ndarray
            Stop indices of valid bouts of data.
        fs : float
            Sampling frequncy in Hz.
        wlen : int
            Window length of the windowed acceleration metric, in seconds.
        cutpoints : dict
            Dictionary of cutpoints.
        acc_metric : numpy.ndarray
            Acceleration metric that has already been windowed and had the
            moving mean computed on for windows of the short window length
            (factor of 60 seconds).

        Returns
        -------
        (ig, ig_itcpt, ig_rsq) : tuple of floats
            The intensity gradient parameters, slope, intercept, and :math:`r^2`.
        names : tuple of strings
            Names of the values.
        """
        hist = zeros(self.ig_vals.size)
        n = int(fs * wlen)

        w_starts = (starts / n).astype(int)
        w_stops = (stops / n).astype(int)

        for start, stop in zip(w_starts, w_stops):
            hist += histogram(acc_metric[start:stop], bins=self.ig_levels, density=False)[0]

        hist /= int(60 / wlen)  # get minutes in each bin

        # get natural log of bin midpoints and accel minutes
        lx = log(self.ig_vals[hist > 0])
        ly = log(hist[hist > 0])

        slope, intercept, rval, *_ = linregress(lx, ly)

        return (slope, intercept, rval**2), self.res_names


class ActivityBoutMinutes(ActivityEndpoint):
    """
    Compute the number of minutes spent in bouts of an activity level. Bouts
    enforces rules about minimum consecutive minutes spent in a certain level.

    Parameters
    ----------
    level : {"MVPA", "sedentary", "light", "moderate", "vigorous"}, optional
        Activity level to compute for. Default is "MVPA".
    bout_length : int, optional
        Bout length in minutes. Default is 5 minutes.
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
    def __init__(self, bout_length=5, bout_criteria=0.8, closed_bout=False, bout_metric=4, **kwargs):
        bout_length = max([int(bout_length), 1])
        bout_criteria = min(max([0.0, bout_criteria]), 1.0)
        if bout_metric not in range(1, 6):
            raise ValueError("bout_metric must be in {1, 2, 3, 4, 5}.")

        super().__init__(
            (
                f"MVPA {bout_length}min bout mins",
                f"Sedentary {bout_length}min bout mins",
                f"Light {bout_length}min bout mins",
                f"Moderate {bout_length}min bout mins",
                f"Vigorous {bout_length}min bout mins",
            ),
            __name__,
        )

        self.bout_len = bout_length
        self.bout_crit = bout_criteria
        self.closed_bout = closed_bout
        self.bout_metric = bout_metric

    def compute(self, starts, stops, fs, wlen, cutpoints, acc_metric_unw, acc_metric):
        """
        Compute the number of total minutes spent in an activity level.

        Parameters
        ----------
        starts : numpy.ndarray
            Start indices of valid bouts of data.
        stops : numpy.ndarray
            Stop indices of valid bouts of data.
        fs : float
            Sampling frequncy in Hz.
        wlen : int
            Window length of the windowed acceleration metric, in seconds.
        cutpoints : dict
            Dictionary of cutpoints.
        acc_metric_unw : numpy.ndarray
            Unwindowed/meaned acceleration metric.
        acc_metric : numpy.ndarray
            Acceleration metric that has already been windowed and had the
            moving mean computed on for windows of the short window length
            (factor of 60 seconds).

        Returns
        -------
        level_mins : float
            Number of minutes spent in the specified activity level.
        """
        lthresh, uthresh = get_level_thresholds(self.level, cutpoints)
        n = int(fs * wlen)

        w_starts = (starts / n).astype(int)
        w_stops = (stops / n).astype(int)

        val = 0.0
        for start, stop in zip(w_starts, w_stops):
            val += get_activity_bouts(
                acc_metric[start:stop],
                lthresh,
                uthresh,
                wlen,
                self.bout_len,
                self.bout_crit,
                self.closed_bout,
                self.bout_metric,
            )

        return val


class ActivityEpochMinutes(ActivityEndpoint):
    """
    Compute the number of minutes spent in an activity level, without any
    determination of bouts.

    Parameters
    ----------
    level : {"MVPA", "sedentary", "light", "moderate", "vigorous"}, optional
        Activity level to compute for. Default is "MVPA".
    day_part : {"wake", "sleep"}
        Label for the part of the day. Used only for the name/endpoint label.
        Default is "wake".

    Notes
    -----
    The number of minutes are computed by simply adding up all the windows of
    short window length that are inside the activity level thresholds, and
    converting the number of windows to a minute value.

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
    def __init__(self, level="MVPA", day_part="wake"):
        if level.lower not in ["mvpa", "sedentary", "light", "moderate", "vigorous"]:
            raise ValueError("level not recognized.")
        if day_part not in ["wake", "sleep"]:
            raise ValueError("day_part must be one of {'wake', 'sleep'}.")

        super().__init__(f"{level} epoch {day_part} mins", __name__)
        self.level = level

    def compute(self, starts, stops, fs, wlen, cutpoints, acc_metric_unw, acc_metric):
        """
        Compute the number of total minutes spent in an activity level.

        Parameters
        ----------
        starts : numpy.ndarray
            Start indices of valid bouts of data.
        stops : numpy.ndarray
            Stop indices of valid bouts of data.
        fs : float
            Sampling frequncy in Hz.
        wlen : int
            Window length of the windowed acceleration metric, in seconds.
        cutpoints : dict
            Dictionary of cutpoints.
        acc_metric_unw : numpy.ndarray
            Unwindowed/meaned acceleration metric.
        acc_metric : numpy.ndarray
            Acceleration metric that has already been windowed and had the
            moving mean computed on for windows of the short window length
            (factor of 60 seconds).

        Returns
        -------
        level_mins : float
            Number of minutes spent in the specified activity level.
        """
        lthresh, uthresh = get_level_thresholds(self.level, cutpoints)
        epm = int(60 / wlen)
        n = int(fs * wlen)

        w_starts = (starts / n).astype(int)
        w_stops = (stops / n).astype(int)

        val = 0.0
        for start, stop in zip(w_starts, w_stops):
            val += sum((acc_metric[start:stop] >= lthresh) & (acc_metric[start:stop] < uthresh)) / epm

        return val / epm


class MaximumAcceleration(ActivityEndpoint):
    """
    Compute the maximum observed mean acceleration over the windows or blocks
    of acceleration data.

    Parameters
    ----------
    block_min : int, optional
        Minutes in each block/window of data. Default is 15 minutes.

    Notes
    -----
    This is computed by taking the moving of the acceleration with windows
    of length `block_min` and finding the window with the maximum mean value.
    """
    def __init__(self, block_min=15):
        block_min = int(block_min)
        super().__init__(f"Max acc {block_min}min blocks gs", __name__)

        self.block_min = block_min

    def compute(self, starts, stops, fs, acc_metric_unw=None, acc_metric=None):
        """
        Compute the maximum acceleration in the specified windows.

        Parameters
        ----------
        starts : numpy.ndarray
            Start indices of valid bouts of data.
        stops : numpy.ndarray
            Stop indices of valid bouts of data.
        fs : float
            Sampling frequncy in Hz.
        acc_metric_unw : {None, numpy.ndarray}, optional
            Unwindowed/meaned acceleration metric.
        acc_metric : {None, numpy.ndarray}, optional
            Acceleration metric that has already been windowed and had the
            moving mean computed on for windows of the short window length
            (factor of 60 seconds).

        Returns
        -------
        max_acc : float
            Maximum acceleration window mean, in units of g.
        """
        super().compute()

        max_acc = 0.0
        nw = int(fs * 60 * self.block_min)
        for s, e in zip(starts, stops):
            res = max(moving_mean(acc_metric_unw[s:e], nw, nw))
            max_acc = max([res, max_acc])

        return max_acc

