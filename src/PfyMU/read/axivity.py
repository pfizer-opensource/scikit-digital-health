"""
Axivity reading functions

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn

from numpy import vstack, insert, append

from PfyMU.base import _BaseProcess
from PfyMU.read._extensions import read_cwa


class ReadCWA(_BaseProcess):
    def __init__(self, base=None, period=None):
        """
        Read a binary CWA file from an axivity sensor into memory. The units for acceleration data
        will be `g`, and the units for angular velocity data will be `deg/s`. If providing a base
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
        ========
        >>> # setup a reader to read, with windows from 08:00 to 20:00 (8AM to 8PM)
        >>> reader = ReadCWA(base=8, period=12)
        >>> reader.predict('example.cwa')
        """
        super().__init__('Read CWA')

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

    def _predict(self, *, file=None, **kwargs):
        """
        Read the data from the axivity file

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be converted by
            `str(file)`
        """
        super()._predict(file=file, **kwargs)

        if file is None:
            raise ValueError("file must not be None")
        if not isinstance(file, str):
            file = str(file)
        if file[-3:] != "cwa":
            warn("File extension is not '.cwa'", UserWarning)

        # read the file
        meta, imudata, ts, idx, light = read_cwa(file, self.base, self.period)

        num_axes = imudata.shape[1]
        gyr_axes = mag_axes = None
        if num_axes == 3:
            acc_axes = slice(None)
        elif num_axes == 6:
            gyr_axes = slice(3)
            acc_axes = slice(3, 6)
        elif num_axes == 9:
            gyr_axes = slice(3)
            acc_axes = slice(3, 6)
            mag_axes = slice(6, 9)
        else:
            raise ValueError("Unexpected number of axes in the IMU data")

        results = {
            self._time: ts
        }
        if acc_axes is not None:
            results[self._acc] = imudata[:, acc_axes]
        if gyr_axes is not None:
            results[self._gyro] = imudata[:, gyr_axes]
        if mag_axes is not None:
            results[self._mag] = imudata[:, mag_axes]

        if self.window:
            day_starts, day_stops = self._get_window_start_stop(idx, ts.size)
            results['day_ends'] = vstack((day_starts, day_stops)).T

    @staticmethod
    def _get_window_start_stop(indices, n_samples):
        """
        Get the correct starts and stops of windows from indices from the GeneActiv reader

        Parameters
        ----------
        indices : numpy.ndarray
            (N, ) array of indices, where N is the number of pages in the file. If the time
            for that page doesn't overlap with either the base/period of a window, then its
            value is set to `-2*expected_num_samples`.
        n_samples : int
            Number of samples in the data read from the GeneActiv file.

        Returns
        -------
        starts : numpy.ndarray
            (M, ) array of the start indices of windows
        stops : numpy.ndarray
            (M, ) array of the stop indices of windows.
        """
        # reader saves indices corresponding to base hour as positive
        base_mask = indices > 0
        # indices correspondingto base + period hour are saved as negative
        period_mask = (indices < 0) & (indices > -(n_samples + 1))

        # temp indices
        start_ = indices[base_mask]
        stop_ = -indices[period_mask]

        if stop_.size == 0:  # only base indices are saved if period is 24h
            starts = insert(start_, 0, 0)
            stops = append(start_, n_samples)
        elif stop_[0] > start_[0]:  # data starts before the first full day
            starts = start_
            if stop_[-1] > start_[-1]:  # data ends after last full day
                stops = stop_
            else:
                stops = append(stop_, n_samples)
        else:  # data stars in the middle of a window
            starts = insert(start_, 0, 0)
            if stop_[-1] > start_[-1]:
                stops = stop_
            else:
                stops = append(stop_, n_samples)

        return starts, stops

