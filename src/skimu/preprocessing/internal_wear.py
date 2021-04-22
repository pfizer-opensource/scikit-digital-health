"""
Internal wear detection to match SleepPy

Pfizer DMTI
2019-2021
"""
from numpy import mean, diff, roll, floor, full, nan, concatenate

from skimu.base import _BaseProcess
from skimu.utility import moving_median, moving_mean, moving_sd
from skimu.utility.internal import rle


class InternalDetectWear(_BaseProcess):
    def __init__(self, temp_thresh=25.0, move_nonwear_thresh=0.0010):
        super().__init__(
            temp_thresh=temp_thresh,
            move_nonwear_thresh=move_nonwear_thresh
        )

        self.temp_thresh = temp_thresh
        self.move_nw_thresh = move_nonwear_thresh

    def predict(self, time=None, accel=None, temperature=None, *, fs=None, **kwargs):
        if fs is None:
            fs = mean(diff(time[:5000]))

        n5 = int(5 * fs)
        shift = int(floor(n5 / 2))
        # 5s rolling median. Roll to replicate centered
        rmd5_acc = roll(moving_median(accel, n5, 1, axis=0, pad=True), shift, axis=0)
        rmd5_temp = roll(moving_median(temperature, n5, 1, pad=True), shift)

        # 5s mean, non-overlapping. These might be 1 shorter than pandas
        rmn5_acc = moving_mean(rmd5_acc, n5, n5, axis=0)
        rmn5_temp = moving_mean(rmd5_temp, n5, n5)

        # 5min median.  Roll to move to center
        rmd_temp = roll(moving_median(rmn5_temp, 60, 1, pad=True), 30)
        # 30min stdev.
        rsd_acc = full((rmd_temp.size, 3), nan)
        rsd_acc[180:-179] = moving_sd(rmn5_acc, 360, 1, axis=0, return_previous=False)

        # nonwear
        temp_nonwear = rmd_temp <= self.temp_thresh
        acc_nonwear = (rsd_acc <= self.move_nw_thresh).any(axis=1)

        nonwear = temp_nonwear | acc_nonwear

        lengths, starts, values = rle(nonwear)
        # get the values for wear times
        lengths = lengths[values == 0]
        starts = starts[values == 0]

        wear = concatenate((starts, starts + lengths)).reshape((2, -1)).T * int(fs * 5)

        # make sure it isn't longer than it can be
        wear[wear > (time.size - 1)] = time.size - 1

        kwargs.update({self._time: time, self._acc: accel, "wear": wear, "fs": fs})
        if self._in_pipeline:
            return kwargs, None
        else:
            return kwargs
