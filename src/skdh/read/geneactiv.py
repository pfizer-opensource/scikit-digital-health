"""
GeneActiv reading process

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn

from numpy import vstack, asarray, int_

from skdh.base import _BaseProcess
from skdh.read._extensions import read_geneactiv


class ReadBin(_BaseProcess):
    """
    Read a binary .bin file from a GeneActiv sensor into memory. Acceleration values are returned
    in units of `g`. If providing a base and period value, included in the output will be the
    indices to create windows starting at the `base` time, with a length of `period`.

    Parameters
    ----------
    bases : {None, int, list-like}, optional
        Base hours [0, 23] in which to start a window of time. Default is None, which will not
        do any windowing. Both `base` and `period` must be defined in order to window. Can use
        multiple, but the number of `bases` must match the number of `periods`.
    periods : {None, int, list-like}, optional
        Periods for each window, in [1, 24]. Defines the number of hours per window. Default is
        None, which will do no windowing. Both `period` and `base` must be defined to window. Can
        use multiple but the number of `periods` must math the number of `bases`.

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

    def __init__(self, bases=None, periods=None):
        super().__init__(
            # kwargs
            bases=bases,
            periods=periods,
        )

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

        Notes
        -----
        The keys in `data` depend on which data the file contained. Potential keys are:

        - `accel`: acceleration [g]
        - `time`: timestamps [s]
        - `light`: light values [unknown]
        - `temperature`: temperature [deg C]
        - `day_ends`: window indices
        """
        if file is None:
            raise ValueError("file must not be None")
        if not isinstance(file, str):
            file = str(file)
        if file[-3:] != "bin":
            warn("File extension is not expected '.bin'", UserWarning)

        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        # read the file
        nmax, fs, acc, time, light, temp, starts, stops = read_geneactiv(
            file, self.bases, self.periods
        )

        results = {
            self._time: time,
            self._acc: acc,
            self._temp: temp,
            "light": light,
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
        if self._in_pipeline:
            return kwargs, None
        else:
            return kwargs
