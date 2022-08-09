"""
GeneActiv reading process

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from warnings import warn

from numpy import vstack, asarray, int_

from skdh.base import BaseProcess
from skdh.io.base import check_input_file
from skdh.io._extensions import read_geneactiv


class ReadBin(BaseProcess):
    """
    Read a binary .bin file from a GeneActiv sensor into memory. Acceleration values are returned
    in units of `g`. If providing a base and period value, included in the output will be the
    indices to create windows starting at the `base` time, with a length of `period`.

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
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.bin).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.

    Examples
    ========
    Setup a reader with no windowing:

    >>> reader = ReadBin()
    >>> reader.predict('example.bin')
    {'accel': ..., 'time': ...}

    Setup a reader that does windowing between 8:00 AM and 8:00 PM (20:00):

    >>> reader = ReadBin(bases=8, periods=12)  # 8 + 12 = 20
    >>> reader.predict('example.bin')
    {'accel': ..., 'time': ..., 'day_ends': [130, 13951, ...]}
    """

    def __init__(self, bases=None, periods=None, ext_error="warn"):
        super().__init__(
            # kwargs
            bases=bases,
            periods=periods,
            ext_error=ext_error,
        )

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

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

    @check_input_file(".bin")
    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from the GeneActiv file

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be converted by
            `str(file)`

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
        - `light`: light values [unknown]
        - `temperature`: temperature [deg C]
        - `day_ends`: window indices
        """
        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        # read the file
        n_max, fs, acc, time, light, temp, starts, stops = read_geneactiv(
            file, self.bases, self.periods
        )

        results = {
            self._time: time[:n_max],
            self._acc: acc[:n_max, :],
            self._temp: temp[:n_max],
            "light": light[:n_max],
            "fs": fs,
            "file": file,
        }

        if self.window:
            results[self._days] = {}
            for i, data in enumerate(zip(self.bases, self.periods)):
                strt = starts[stops[:, i] != 0, i]
                stp = stops[stops[:, i] != 0, i]

                results[self._days][(data[0], data[1])] = vstack((strt, stp)).T

        kwargs.update(results)

        return (kwargs, None) if self._in_pipeline else kwargs
