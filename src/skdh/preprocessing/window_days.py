"""
Functions to apply to data post-reading.

Lukas Adamowicz
Copyright (c) 2024. Pfizer Inc. All rights reserved.
"""

from warnings import warn

from numpy import asarray, int_, mean, diff, abs
from pandas import DateOffset

from skdh.base import BaseProcess, handle_process_returns


def finalize_guess(time, fs, guess, target):
    """
    Get the final guess for the index of a window start or stop

    Parameters
    ----------
    time : numpy.ndarray
        Array of unix timestamps
    fs : float
        Sampling frequency in Hz.
    guess : int
        Guess for the index of the window start/end.
    target : pandas.Timestamp
        Target time for the window start/end.

    Returns
    -------
    final_i : int
        Index of the window start/end.
    """
    i1 = max(guess - 1, 0)
    i2 = guess
    i3 = min(guess + 1, time.size - 1)

    ts_target = target.timestamp()

    if i2 <= 0:
        return 0
    if i2 >= (time.size - 1):
        return time.size - 1

    check1 = abs(time[i2] - ts_target) <= abs(time[i1] - ts_target)
    check3 = abs(time[i2] - ts_target) <= abs(time[i3] - ts_target)

    # path 1: guess is smallest value
    if check1 and check3:
        return i2
    elif not check1:  # path 2: smaller value to the left side
        # make an intelligent update to guess to avoid recursion limits (avoiding if statement)
        guess = (
            guess
            - 1
            + (int((time[i2] - ts_target) > 5) * int((ts_target - time[i1]) * fs))
        )
    elif not check3:  # path 3: smaller value to the right side
        # make an intelligent update to guess to avoid recursion limits (avoiding if statement)
        guess = (
            guess
            + 1
            + (int((ts_target - time[i2]) > 5) * int((ts_target - time[i3]) * fs))
        )

    return finalize_guess(time, fs, guess, target)


class GetDayWindowIndices(BaseProcess):
    """
    Get the indices corresponding to days.

    Parameters
    ----------
    bases : {None, int, list-like}, optional
        Base hours [0, 23] in which to start a window of time. Default is None,
        which will not do any windowing. Both `base` and `period` must be defined
        in order to window. Can use multiple, but the number of `bases` must match
        the number of `periods`.
    periods : {None, int, list-like}, optional
        Periods for each window, in [1, 24]. Defines the number of hours per window.
        Default is None, which will do no windowing. Both `period` and `base` must
        be defined to window. Can use multiple but the number of `periods` must
        match the number of `bases`.
    """

    def __init__(self, bases=None, periods=None):
        super().__init__(bases=bases, periods=periods)

        if (bases is None) and (periods is None):
            self.window = False
            self.bases = asarray([0])  # needs to be defined for passing to extensions
            self.periods = asarray([12])
        elif (bases is None) or (periods is None):
            warn("One of base or period is None, not windowing", UserWarning)
            self.window = False
            self.bases = asarray([0])
            self.periods = asarray([12])
        else:
            if isinstance(bases, int) and isinstance(periods, int):
                bases = asarray([bases])
                periods = asarray([periods])
            else:
                bases = asarray(bases, dtype=int_)
                periods = asarray(periods, dtype=int_)

            if ((0 <= bases) & (bases <= 23)).all() and (
                (1 <= periods) & (periods <= 24)
            ).all():
                self.window = True
                self.bases = bases
                self.periods = periods
            else:
                raise ValueError(
                    "Base must be in [0, 23] and period must be in [1, 23]"
                )

    def window_days(self, time, fs):
        """
        Get the indices for days based on the windowing parameters.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        fs : float, optional
            Sampling frequency in Hz. If not provided, it is calculated from the
            timestamps.

        Returns
        -------
        days : dict
            Dictionary of day starts and ends
        """
        dt0 = self.convert_timestamps(time[0])

        day_delta = DateOffset(days=1)

        # approximate number of samples per day
        samples_per_day = int(86400 * fs)

        # find the end time for windows
        w_ends = (asarray(self.bases) + asarray(self.periods)) % 24

        # iterate over the bases and periods
        day_windows = {}
        for start, end, p in zip(self.bases, w_ends, self.periods):
            # get the start and end times
            start_time = dt0.replace(hour=start, minute=0, second=0, microsecond=0)
            end_time = dt0.replace(hour=end, minute=0, second=0, microsecond=0)

            # only subtract a day if the end time is after the start time
            # this works because we added the base to the period and took the mod 24
            # e.g.
            # base = 12, period = 4 -> end = 16, need to subtract a day from both to keep period=4
            # base = 12, period = 20 -> end = 8, don't need to subtract a day to keep period=20
            if end_time > start_time:
                end_time -= day_delta
            # do this second so that we can do the above check first
            start_time -= day_delta

            # make sure that at least one of the timestamps is during the recording
            while end_time <= dt0:
                start_time += day_delta
                end_time += day_delta

            # create a first guess for the indices
            guess_start = int((start_time.timestamp() - time[0]) * fs)
            guess_end = int((end_time.timestamp() - time[0]) * fs)

            windows = []
            while start_time.timestamp() < time[-1]:
                # finalize the guess and append to windows
                windows.append(
                    [
                        finalize_guess(time, fs, guess_start, start_time),
                        finalize_guess(time, fs, guess_end, end_time),
                    ]
                )

                # update the guesses
                guess_start += samples_per_day
                guess_end += samples_per_day

                # update the start/end time
                start_time += day_delta
                end_time += day_delta
            day_windows[(start, p)] = asarray(windows)

        return day_windows

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, *, time, fs=None, tz_name=None, **kwargs):
        """
        predict(*, time, fs=None)

        Compute the indices for days.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        fs : float, optional
            Sampling frequency in Hz. If not provided, it is calculated from the
            timestamps.
        tz_name : {None, str}, optional
            Timezone name. If none provided, timestamps are assumed to be naive
            in local time.

        Returns
        -------
        data : dict
            Dictionary with the key `day_ends`, which itself is a dictionary
            of (N, 2) indices corresponding to [start, stop] of days. Keys are
            tuples of `(base, period)`.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            fs=fs,
            tz_name=tz_name,
            **kwargs,
        )

        if not self.window:
            return {}

        # calculate fs if necessary
        fs = 1 / mean(diff(time)) if fs is None else fs

        days = self.window_days(time, fs)

        return {self._days: days}
