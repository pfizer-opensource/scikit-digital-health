"""
Axivity reading functions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from numpy import ascontiguousarray

from skdh.base import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file, handle_naive_timestamps
from skdh.io._extensions import read_axivity


class UnexpectedAxesError(Exception):
    pass


class ReadCwa(BaseProcess):
    """
    Read a binary CWA file from an axivity sensor into memory. Acceleration is return in units of
    'g' while angular velocity (if available) is returned in units of `deg/s`.

    Parameters
    ----------
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.cwa).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.

    .. deprecated:: 0.14.0
        `bases` Removed in favor of having windowing be its own class,
        :class:`skdh.preprocessing.GetDayWindowIndices`.
        `periods` Removed in favor of having windowing be its own class.

    Examples
    --------
    Setup a reader:

    >>> reader = ReadCwa()
    >>> reader.predict('example.cwa')
    {'accel': ..., 'time': ..., ...}
    """

    def __init__(self, *, ext_error="warn"):
        super().__init__(
            # kwargs
            ext_error=ext_error,
        )

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(".cwa")
    def predict(self, *, file, tz_name=None, **kwargs):
        """
        predict(*, file)

        Read the data from the axivity file

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be converted by
            `str(file)`.
        tz_name : {None, str}, optional
            IANA time-zone name for the recording location. If not provided, timestamps
            will represent local time naively. This means they will not account for
            any time changes due to Daylight Saving Time.

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
        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs
        )

        # read the file
        fs, n_bad_samples, imudata, ts, temperature = read_axivity(str(file))

        # end = None if n_bad_samples == 0 else -n_bad_samples
        end = None

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
            self._time: handle_naive_timestamps(
                ts[:end], is_local=True, tz_name=tz_name
            ),
            "fs": fs,
            self._temp: temperature[:end],
        }
        if acc_axes is not None:
            results[self._acc] = ascontiguousarray(imudata[:end, acc_axes])
        if gyr_axes is not None:
            results[self._gyro] = ascontiguousarray(imudata[:end, gyr_axes])
        if mag_axes is not None:  # pragma: no cover :: don't have data to test this
            results[self._mag] = ascontiguousarray(imudata[:end, mag_axes])

        return results
