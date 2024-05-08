"""
Functions to apply to data post-reading.

Lukas Adamowicz
Copyright (c) 2024. Pfizer Inc. All rights reserved.
"""

from warnings import warn

from numpy import asarray, int_, mean, diff

from skdh.base import BaseProcess, handle_process_returns
from skdh.preprocessing._extensions import cwindow_days


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

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, *, time, fs=None, **kwargs):
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

        Returns
        -------
        data : dict
            Dictionary with the key `day_ends`, which itself is a dictionary
            of (N, 2) indices corresponding to [start, stop] of days. Keys are
            tuples of `(base, period)`.
        """
        super().predict(
            expect_days=False, expect_wear=False, time=time, fs=fs, **kwargs
        )

        if not self.window:
            return {}

        # calculate fs if necessary
        fs = 1 / mean(diff(time)) if fs is None else fs

        # get the indices
        raw_days = cwindow_days(time, fs, self.bases, self.periods)

        # create the return dictionary
        days = {}
        for i, (b, p) in enumerate(zip(self.bases, self.periods)):
            # filter out extra indices, which will be indicated by both
            # start and stop being 0
            mask = (raw_days[i, :, 0] == 0) & (raw_days[i, :, 1] == 0)

            days[(b, p)] = raw_days[i][~mask, :]

        return {self._days: days}
