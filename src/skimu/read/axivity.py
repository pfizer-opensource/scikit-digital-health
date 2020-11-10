"""
Axivity reading functions

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn

from numpy import vstack

from skimu.base import _BaseProcess
from skimu.read.get_window_start_stop import get_window_start_stop
from skimu.read._extensions import read_cwa


class UnexpectedAxesError(Exception):
    pass


class ReadCWA(_BaseProcess):
    """
    Read a binary CWA file from an axivity sensor into memory. Acceleration is return in units of
    'g' while angular velocity (if available) is returned in units of `deg/s`. If providing a base
    and period value, included in the output will be the indices to create windows starting at
    the `base` time, with a length of `period`.

    Parameters
    ----------
    base : {None, int}, optional
        Base hour [0, 23] in which to start a window of time. Default is None, which will not
        do any windowing. Both `base` and `period` must be defined in order to window.
    period : {None, int}, optional
        Period for each window, in [1, 24]. Defines the number of hours per window. Default is
        None, which will do no windowing. Both `period` and `base` must be defined to window

    Examples
    --------
    Setup a reader with no windowing:

    >>> reader = ReadCWA()
    >>> reader.predict('example.cwa')
    {'accel': ..., 'time': ..., ...}

    Setup a reader that does windowing between 8:00 AM and 8:00 PM (20:00):

    >>> reader = ReadCWA(base=8, period=12)  # 8 + 12 = 20
    >>> reader.predict('example.cwa')
    {'accel': ..., 'time': ..., 'day_ends': [130, 13951, ...], ...}
    """

    def __init__(self, base=None, period=None):
        super().__init__(
            # kwargs
            base=base,
            period=None
        )

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
                raise ValueError("Base must be in [0, 23] and period must be in [1, 23]")

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
        super().predict(file=file, **kwargs)

        if file is None:
            raise ValueError("file must not be None")
        if not isinstance(file, str):
            file = str(file)
        if file[-3:] != "cwa":
            warn("File extension is not expected '.cwa'", UserWarning)

        # read the file
        meta, imudata, ts, idx, light = read_cwa(file, self.base, self.period)

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
            self._time: ts
        }
        if acc_axes is not None:
            results[self._acc] = imudata[:, acc_axes]
        if gyr_axes is not None:
            results[self._gyro] = imudata[:, gyr_axes]
        if mag_axes is not None:  # pragma: no cover :: don't have data to test this
            results[self._mag] = imudata[:, mag_axes]

        if self.window:
            day_starts, day_stops = get_window_start_stop(idx, ts.size)
            results[self._days] = vstack((day_starts, day_stops)).T

        kwargs.update(results)
        if self._in_pipeline:
            return kwargs, None
        else:
            return kwargs
