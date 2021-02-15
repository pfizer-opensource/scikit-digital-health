"""
Wear detection algorithms

Lukas Adamowicz
Pfizer DMTI 2021
"""
from warnings import warn

from numpy import mean, diff, sum, insert, append, nonzero, delete, concatenate, unique, int_

from skimu.base import _BaseProcess
from skimu.utility import rolling_sd, get_windowed_view


class DetectWear(_BaseProcess):
    """
    Detect periods of non-wear in accelerometer recordings.

    Parameters
    ----------
    sd_crit : float, optional
        Acceleration standard deviation threshold for determining non-wear. Default is 0.013, which
        was observed for GeneActiv devices during motionless bench tests, and will likely depend
        on the brand of accelerometer being used.
    range_crit : float, optional
        Acceleration window range threshold for determining non-wear. Default is 0.050.
    window_length : int, optional
        Number of minutes in a window used to determine non-wear. Default is 60 minutes.
    window_skip : int, optional
        Number of minutes to skip between windows. Default is 15 minutes, which would result
        in window overlaps of 45 minutes with the default 60 minute `window_length`.


    References
    ----------
    .. [1] V. T. van Hees et al., “Separating Movement and Gravity Components in an Acceleration
        Signal and Implications for the Assessment of Human Daily Physical Activity,” PLOS ONE,
        vol. 8, no. 4, p. e61691, Apr. 2013, doi: 10.1371/journal.pone.0061691.
    """
    def __init__(self, sd_crit=0.013, range_crit=0.050, window_length=60, window_skip=15):
        window_length = int(window_length)
        window_skip = int(window_skip)
        super().__init__(
            sd_crit=sd_crit,
            range_crit=range_crit,
            window_length=window_length,
            window_skip=window_skip
        )

        self.sd_crit = sd_crit
        self.range_crit = range_crit
        self.wlen = window_length
        self.wskip = window_skip

    def predict(self, time=None, accel=None, **kwargs):
        """
        Detect the periods of non-wear

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps (in seconds) since 1970-01-01.
        accel : numpy.ndarray
            (N, 3) array of measured acceleration values in units of g.

        Returns
        -------
        results : dictionary
            Dictionary of inputs, plus the key `wear` which is a (N, ) boolean array where `True`
            indicates the accelerometer was worn.
        """
        # dont start at zero due to timestamp weirdness with some devices
        fs = 1 / mean(diff(time[1000:5000]))
        n_wlen = int(self.wlen * 60 * fs)  # samples in wlen minutes
        n_wskip = int(self.wskip * 60 * fs)  # samples in wskip minutes

        # note that while this block starts at 0, the method uses centered blocks, which
        # means that the first block actually corresponds to a block starting 22.5 minutes into
        # the recording
        acc_rsd = rolling_sd(accel, n_wlen, n_wskip, axis=0, return_previous=False)

        # get the accelerometer range in each 60min window
        acc_w = get_windowed_view(accel, n_wlen, n_wskip)
        acc_w_range = acc_w.max(axis=1) - acc_w.min(axis=1)

        nonwear = sum((acc_rsd < self.sd_crit) & (acc_w_range < 0.050), axis=1) >= 2

        # flip to wear starts/stops now
        wear_start = nonzero(diff(nonwear.astype(int_)) == -1)[0] + 1
        wear_stop = nonzero(diff(nonwear.astype(int_)) == 1)[0] + 1

        if (wear_stop[0] < wear_start[0]) & (not nonwear[0]):
            wear_start = insert(wear_start, 0, 0)
        else:
            warn("Non-wear periods have incompatible starts/stops. Skipping nonwear detection.")
            kwargs.update({self._time: time, self._acc: accel})
            return kwargs

        if (wear_stop[-1] < wear_start[-1]) & (not nonwear[-1]):
            wear_stop = append(wear_stop, nonwear.size - 1)
        else:
            warn("Non-wear periods have incompatible starts/stops. Skipping nonwear detection.")
            kwargs.update({self._time: time, self._acc: accel})
            return kwargs

        wear_time = (wear_stop - wear_start) * (self.wskip / 60)  # in hours
        nonwear_times = (wear_start[1:] - wear_stop[:-1]) * (self.wskip / 60)  # in hours

        # compute the wear time as a percentage of the sum of surrounding non-wear time
        perc_time = wear_time[1:-1] / (nonwear_times[:-1] + nonwear_times[1:])

        # filter out wear times less than 6 hours comprising less than 30% of surrounding non-wear
        wt6 = nonzero(wear_time[1:-1] <= 6)
        switch6 = wt6[perc_time[wt6] <= 0.3]

        # filter out wear times less than 3 hours comprising less than 80% of surrounding non-wear
        wt3 = nonzero(wear_time[1:-1] <= 3)
        switch3 = wt3[perc_time[wt3] <= 0.8]


def _modify_wear_times(nonwear, wskip):
    nw_start = w_stop = nonzero(diff(nonwear.astype(int_)) == 1)[0] + 1
    nw_stop = w_start = nonzero(diff(nonwear.astype(int_)) == -1)[0] + 1

    if nonwear[0]:
        nw_start = insert(nw_start, 0, 0)
    else:
        w_start = insert(w_start, 0, 0)

    if nonwear[-1]:
        nw_stop = append(nw_stop, nonwear.size)
    else:
        w_stop = append(w_stop, nonwear.size)

    nw_times = (nw_stop - nw_start) * (wskip / 60)
    w_times = (w_stop - w_start) * (wskip / 60)  # in hours

    # 3 different paths based on times length
    if nw_times.size == w_times.size:  # [NW][W][NW][W] or [W][NW][W][NW]
        if w_start[0] < nw_start[0]:
            idx = slice(1, None, None)
        else:
            idx = slice(None, -1, None)
    elif nw_times.size == w_times.size - 1:  # [W][NW][W][NW][W]
        idx = slice(1, -1, None)
    elif nw_times.size == w_times.size + 1:  # [NW][W][NW][W][NW]
        idx = slice(None, None, None)
    else:
        warn("Wear/non-wear periods are not correct, skipping...", UserWarning)
        return None, None

    pct = w_times[idx] / (nw_times[:-1] + nw_times[1:])
    wt6 = nonzero(w_times[idx] <= 6)[0]
    wt3 = nonzero(w_times[idx] <= 3)[0]

    switch6 = wt6[pct[wt6] < 0.3]
    switch3 = wt3[pct[wt3] < 0.8]

    switch = unique(concatenate((switch6, switch3))) + idx.indices(3)[0]  # start is always under 3
    w_start = delete(w_start, switch)
    w_stop = delete(w_stop, switch)

    return w_start, w_stop
