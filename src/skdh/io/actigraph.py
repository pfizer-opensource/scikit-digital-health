"""
Actigraph reading functions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from warnings import warn

from numpy import vstack, allclose

from skdh.base import BaseProcess
from skdh.io.base import check_input_file
from skdh.io.get_window_start_stop import get_window_start_stop
from skdh.io._extensions import read_gt3x


class ReadGT3X(BaseProcess):
    """
    Read a GT3X archive file from an Actigraph sensor into memory. Acceleration
    is returned in units of 'g', while time is unix time in seconds. If providing
    a base and period value, included in the output will be the indices to create
    windows starting at the `base` time, with a length of `period` hours.  If the
    GT3X file was produced by an older version (wearable with a light sensor), the
    lux values returned will be useful.

    Parameters
    ----------
    base : {None, int}, optional
        Base hour [0, 23] in which to start a window of time. Default is None, which
        will not do any windowing. Both `base` and `period` must be defined in order
        to window.
    period : {None, int}, optional
        Period for each window, in [1, 24]. Defines the number of hours per window.
        Default is None, which will do no windowing. Both `period` and `base` must be
        defined to window.
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.gt3x).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.

    Warnings
    --------
    Due to continuous changes in GT3X file structures, this functionality is
    provided AS-IS. Users should validate the output for their version of files
    with the ActiGraph CSV export.

    Examples
    --------
    Setup a reader with no windowing:

    >>> reader = ReadGT3X()
    >>> reader.predict('example.gt3x')
    {'accel': ..., 'time': ..., ...}

    Setup a reader that does windowing between 8:00 AM and 8:00 PM (20:00):

    >>> reader = ReadGT3X(base=8, period=12)  # 8 + 12 = 20
    >>> reader.predict('example.gt3x')
    {'accel': ..., 'time': ..., 'day_ends': [130, 13950, ...], ...}
    """

    def __init__(self, base=None, period=None, ext_error="warn"):
        super().__init__(
            # kwargs
            base=base,
            period=None,
            ext_error=ext_error,
        )

        warn(
            "This class is provided as-is. Validate output compared to ActiGraph CSV.",
            UserWarning,
        )

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

        if (base is None) and (period is None):
            self.window = False
            self.base = 0  # needs to be defined for passing to extensions
            self.period = 12
        elif (base is None) or (period is None):
            warn("One of base or period is None, not windowing", UserWarning)
            self.window = False
            self.base = 0
            self.period = 12
        else:
            if (0 <= base <= 23) and (1 <= period <= 24):
                self.window = True
                self.base = base
                self.period = period
            else:
                raise ValueError(
                    "Base must be in [0, 23] and period must be in [1, 23]"
                )

    @check_input_file(".gt3x")
    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from the GT3X archive file

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be
            converted by `str(file)`

        Returns
        -------
        data : dict
            Dictionary of the data contained in the file.

        Raises
        ------
        ValueError
            If the file name is not provided

        Notes
        -----
        The keys in `data` depend on which data the file contained. Potential keys are:

        - `accel`: acceleration [g]
        - `time`: timestamps [s]
        - `lux`: light readings. Note that this will not be returned if the data is
          not valid
        - `day_ends`: window indices
        """
        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        time, accel, lux, index, N = read_gt3x(file, self.base, self.period)

        results = {self._time: time[:N], self._acc: accel[:N], "file": file}

        # can't include it by default, because if it is garbage data, dont want
        # that influencing downstream results. Better to just not use in that case
        if not allclose(lux, 0.0):
            results["light"] = lux[:N]

        if self.window:
            day_starts, day_stops = get_window_start_stop(index, N)
            results[self._days] = {
                (self.base, self.period): vstack((day_starts, day_stops)).T
            }

        kwargs.update(results)
        return (kwargs, None) if self._in_pipeline else kwargs
