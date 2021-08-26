"""
Activity endpoint definitions

Lukas Adamowicz
Pfizer DMTI 2021
"""
from warnings import warn

from numpy import array, zeros, max, nanmax, histogram, log, nan, sum, nonzero, maximum, argmax, int_, floor, ceil
from scipy.stats import linregress

from skdh.utility import moving_mean
from skdh.activity.cutpoints import _base_cutpoints, get_level_thresholds


__all__ = ["ActivityEndpoint", "IntensityGradient", "MaxAcceleration", "TotalIntensityTime", "BoutIntensityTime"]


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
        try:
            # window determination can go back to left justified
            rm = moving_mean(xt, nboutdur, 1)
        except ValueError:
            return 0.0

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
        try:
            rm = moving_mean(xt, nboutdur, 1)
        except ValueError:
            return 0.0

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
        try:
            rm = moving_mean(xt, nboutdur, 1)
        except ValueError:
            return 0.0

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


class ActivityEndpoint:
    def __init__(self, name, state):
        if isinstance(name, (tuple, list)):
            self.name = [f'{state} {i}' for i in name]
        else:
            self.name = f'{state} {name}'

        self.state = state

    def predict(self, **kwargs):
        pass

    def reset_cached(self):
        pass


class IntensityGradient(ActivityEndpoint):
    """
    Compute the gradient of the acceleration movement intensity.
    """
    def __init__(self, state='wake'):
        super(IntensityGradient, self).__init__(
            ["intensity gradient", 'ig intercept', 'ig r-squared'],
            state
        )

        # default from rowlands
        self.ig_levels = array([i for i in range(0, 4001, 25)] + [8000], dtype="float") / 1000
        self.ig_vals = (self.ig_levels[1:] + self.ig_levels[:-1]) / 2

        # values that need to be cached and stored between runs
        self.hist = zeros(self.ig_vals.size)
        self.ig = None
        self.ig_int = None
        self.ig_r = None
        self.i = None

    def predict(self, results, i, accel_metric, epoch_s, epochs_per_min, **kwargs):
        super(IntensityGradient, self).predict()

        # get the counts in number of minutes in each intensity bin
        self.hist += histogram(accel_metric, bins=self.ig_levels, density=False)[0] / epochs_per_min

        # get pointers to the intensity gradient results
        self.ig = results[self.name[0]]
        self.ig_int = results[self.name[1]]
        self.ig_r = results[self.name[2]]
        self.i = i

    def reset_cached(self):
        super(IntensityGradient, self).reset_cached()

        # make sure we have results locations to set
        if all([i is not None for i in [self.ig, self.ig_int, self.ig_r, self.i]]):
            # compute the results
            # convert back to mg to match existing work
            lx = log(self.ig_vals[self.hist > 0] * 1000)
            ly = log(self.hist[self.hist > 0])

            if ly.size <= 1:
                slope = intercept = rval = nan
            else:
                slope, intercept, rval, *_ = linregress(lx, ly)

            # set the results values
            self.ig[self.i] = slope
            self.ig_int[self.i] = intercept
            self.ig_r[self.i] = rval ** 2

        # reset the histogram counts to 0, and results to None
        self.hist = zeros(self.ig_vals.size)
        self.ig = None
        self.ig_int = None
        self.ig_r = None
        self.i = None


class MaxAcceleration(ActivityEndpoint):
    """
    Compute the maximum acceleration over windows of the specified length.
    """
    def __init__(self, window_lengths, state='wake'):
        if isinstance(window_lengths, int):
            window_lengths = [window_lengths]

        super().__init__(
            [f'max acc {i}min [g]' for i in window_lengths], state
        )

        self.wlens = window_lengths

    def predict(self, results, i, accel_metric, epoch_s, epochs_per_min, **kwargs):
        super(MaxAcceleration, self).predict()

        for wlen, name in zip(self.wlens, self.name):
            n = wlen * epochs_per_min
            # skip 1 sample because we want the window with the largest acceleration
            # skipping more samples would introduce bias by random chance of
            # where the windows start and stop
            try:
                tmp_max = max(moving_mean(accel_metric, n, 1))
            except ValueError:
                return  # if the window length is too long for this block of data

            # check that we don't have a larger result already for this day
            results[name][i] = nanmax([tmp_max, results[name][i]])


class TotalIntensityTime(ActivityEndpoint):
    """
    Compute the total time spent in an intensity level
    """
    def __init__(self, level, epoch_length, cutpoints=None, state='wake'):
        super().__init__(f'{level} {epoch_length}s epoch [min]', state)
        self.level = level

        if cutpoints is None:
            warn(f"Cutpoints not specified for {self!r}. Using `migueles_wrist_adult`")
            cutpoints = _base_cutpoints['migueles_wrist_adult']

        self.lthresh, self.uthresh = get_level_thresholds(self.level, cutpoints)

    def predict(self, results, i, accel_metric, epoch_s, epochs_per_min, **kwargs):
        super().predict()

        time = sum((accel_metric >= self.lthresh) & (accel_metric < self.uthresh))

        results[self.name][i] += time / epochs_per_min


class BoutIntensityTime(ActivityEndpoint):
    """
    Compute the time spent in bouts of intensity levels.
    """
    def __init__(self, level, bout_lengths, bout_criteria, bout_metric, closed_bout, cutpoints=None, state='wake'):
        if isinstance(bout_lengths, int):
            bout_lengths = [bout_lengths]

        super(BoutIntensityTime, self).__init__(
            [f'{level} {i}min bout [min]' for i in bout_lengths],
            state
        )
        self.level = level
        self.blens = bout_lengths
        self.bcrit = bout_criteria
        self.bmetric = bout_metric
        self.cbout = closed_bout

        if cutpoints is None:
            warn(f"Cutpoints not specified for {self!r}. Using `migueles_wrist_adult`")
            cutpoints = _base_cutpoints['migueles_wrist_adult']

        self.lthresh, self.uthresh = get_level_thresholds(self.level, cutpoints)

    def predict(self, results, i, accel_metric, epoch_s, epochs_per_min, **kwargs):
        super().predict()

        for bout_len, name in zip(self.blens, self.name):
            results[name][i] += get_activity_bouts(
                accel_metric,
                self.lthresh,
                self.uthresh,
                epoch_s,
                bout_len,
                self.bcrit,
                self.cbout,
                self.bmetric,
            )
