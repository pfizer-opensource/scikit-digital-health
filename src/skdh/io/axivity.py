"""
Axivity reading functions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from warnings import warn

from numpy import vstack, asarray, ascontiguousarray, minimum, int_

from skdh.base import BaseProcess
from skdh.io.base import check_input_file
from skdh.io._extensions import read_axivity


class UnexpectedAxesError(Exception):
    pass


class ReadCwa(BaseProcess):
    """
    Read a binary CWA file from an axivity sensor into memory. Acceleration is return in units of
    'g' while angular velocity (if available) is returned in units of `deg/s`. If providing a base
    and period value, included in the output will be the indices to create windows starting at
    the `base` time, with a length of `period`.

    Parameters
    ----------
    bases : {None, int}, optional
        Base hour [0, 23] in which to start a window of time. Default is None, which
        will not do any windowing. Both `base` and `period` must be defined in order
        to window.
    periods : {None, int}, optional
        Period for each window, in [1, 24]. Defines the number of hours per window.
        Default is None, which will do no windowing. Both `period` and `base` must
        be defined to window
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.cwa).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.

    Examples
    --------
    Setup a reader with no windowing:

    >>> reader = ReadCwa()
    >>> reader.predict('example.cwa')
    {'accel': ..., 'time': ..., ...}

    Setup a reader that does windowing between 8:00 AM and 8:00 PM (20:00):

    >>> reader = ReadCwa(bases=8, periods=12)  # 8 + 12 = 20
    >>> reader.predict('example.cwa')
    {'accel': ..., 'time': ..., 'day_ends': [130, 13951, ...], ...}
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

    @check_input_file(".cwa")
    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from the axivity file

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
        UnexpectedAxesError
            If the number of axes returned is not 3, 6 or 9

        Notes
        -----
        The keys in `data` depend on which data the file contained. Potential keys are:

        - `accel`: acceleration [g]
        - `gyro`: angular velocity [deg/s]
        - `magnet`: magnetic field readings [uT]
        - `time`: timestamps [s]
        - `day_ends`: window indices
        """
        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        # read the file
        fs, n_bad_samples, imudata, ts, temperature, starts, stops = read_axivity(
            file, self.bases, self.periods
        )

        end = None if n_bad_samples == 0 else -n_bad_samples

        num_axes = imudata.shape[1]
        gyr_axes = mag_axes = None
        if num_axes == 3:
            acc_axes = slice(None)
        elif num_axes == 6:
            gyr_axes = slice(3)
            acc_axes = slice(3, 6)
        elif num_axes == 9:  # pragma: no cover :: don't have data to test this
            gyr_axes = slice(3)
            acc_axes = slice(3, 6)
            mag_axes = slice(6, 9)
        else:  # pragma: no cover :: not expected to reach here only if file is corrupt
            raise UnexpectedAxesError("Unexpected number of axes in the IMU data")

        results = {
            self._time: ts[:end],
            "file": file,
            "fs": fs,
            self._temp: temperature[:end],
        }
        if acc_axes is not None:
            results[self._acc] = ascontiguousarray(imudata[:end, acc_axes])
        if gyr_axes is not None:
            results[self._gyro] = ascontiguousarray(imudata[:end, gyr_axes])
        if mag_axes is not None:  # pragma: no cover :: don't have data to test this
            results[self._mag] = ascontiguousarray(imudata[:end, mag_axes])

        if self.window:
            results[self._days] = {}
            for i, data in enumerate(zip(self.bases, self.periods)):
                strt = starts[stops[:, i] != 0, i]
                stp = stops[stops[:, i] != 0, i]

                results[self._days][(data[0], data[1])] = minimum(
                    vstack((strt, stp)).T, results[self._time].size - 1
                )

        kwargs.update(results)

        return (kwargs, None) if self._in_pipeline else kwargs
