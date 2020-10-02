"""
GeneActiv reading process

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn

from numpy import vstack

from PfyMU.base import _BaseProcess
from PfyMU.read.get_window_start_stop import get_window_start_stop
from PfyMU.read._extensions import bin_convert


class ReadBin(_BaseProcess):
    def __repr__(self):
        return f"ReadBin(base={self.base!r}, period={self.period!r})"

    def __init__(self, base=None, period=None):
        """
        Read a binary .bin file from a GeneActiv sensor into memory. The units for acceleration
        data will be `g`. If providing a base and period value, included in the output will be the
        indices to create windows starting at the `base` time, with a length of `period`.

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
        >>> reader = ReadBin(base=8, period=12)
        >>> reader.predict('example.bin')
        """
        super().__init__('Read bin', False)

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

    def predict(self, *args, **kwargs):
        """
        Read the data from the axivity file

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be converted by
            `str(file)`
        """
        super().predict(*args, **kwargs)

    def _predict(self, file=None, **kwargs):
        if file is None:
            raise ValueError("file must not be None")
        if not isinstance(file, str):
            file = str(file)
        if file[-3:] != "bin":
            warn("File extension is not '.bin'", UserWarning)

        # read the file
        nmax, hdr, acc, time, light, temp, idx, dtime = bin_convert(file, self.base, self.period)

        results = {
            self._time: time,
            self._acc: acc,
            self._temp: temp,
            'light': light
        }

        if self.window:
            day_starts, day_stops = get_window_start_stop(idx, time.size)
            results[self._days] = vstack((day_starts, day_stops)).T

        kwargs.update(results)
        return kwargs, None
