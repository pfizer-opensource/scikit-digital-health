"""
Base classes, functions, etc for the skimu library

Lukas Adamowicz
Pfizer DMTI 2020
"""
from datetime import date as dt_date
import logging
from pathlib import Path

from pandas import DataFrame
from numpy import array


class _BaseProcess:
    # names of the variables that are passed to predict
    # CHANGE IF predict/_predict call changes!
    _file = "file"
    _time = "time"
    _acc = "accel"
    _gyro = "gyro"
    _mag = "magnet"
    _temp = "temperature"
    _days = "day_ends"

    def __str__(self):
        return self._name

    def __repr__(self):
        ret = f"{self._name}("
        for k in self._kw:
            ret += f"{k}={self._kw[k]!r}, "
        ret = ret[:-2] + ")"

        return ret

    def __init__(self, **kwargs):
        """
        Intended to be subclassed

        Parameters
        ----------
        return_result : bool, optional
            Return the result part of predict, or the input/output dictionary
        kwargs
            Key-word arguments which are passed to the sub-class
        """
        self._name = self.__class__.__name__
        self._in_pipeline = False  # initialize to false.  Will be set by the pipeline

        self._kw = kwargs

        self.logger = logging.getLogger(__name__)

        # default day key
        self.day_key = (-1, -1)

        # day and wear indices
        self.day_idx = (None, None)
        self.wear_idx = (None, None)

        # for plotting
        self.f = self.ax = self.plot_fname = None

    def _check_if_idx_none(self, var, msg_if_none, i1, i2):
        """
        Checks if an index value is None. Returns set indices if it is
        or the starts and stops if it is not.

        Parameters
        ----------
        var : {None, numpy.ndarray}
            None, or a 2d ndarray of start and stop indices shape(N, 2).
        msg_if_none : str
            Message to log if the `var` is None.
        i1 : {None, int}
            Start index if `var` is None.
        i2 : {None, int}
            Stop index if `var` is None.

        Returns
        -------
        starts : numpy.ndarray
            ndarray of start indices.
        stops : numpy.ndarray
            ndarray of stop indices.
        """
        if var is None:
            self.logger.info(msg_if_none)
            if i1 is None or i2 is None:
                return None, None
            else:
                start, stop = array([i1]), array([i2])
        else:
            start, stop = var[:, 0], var[:, 1]

        return start, stop

    def predict(self, *args, **kwargs):
        """
        Intended to be overwritten in the subclass. Should still be called
        with super.
        """
        self.logger.info(f"Entering {self._name} processing with call {self!r}")
        # save the filename for saving reference
        self._file_name = Path(kwargs.get("file", "")).stem

        # TODO if the accel/time requirement for some things changes then this
        # will need to change as well
        n = kwargs.get(self._acc, kwargs.get(self._time)).shape[0] - 1

        days = kwargs.get(self._days, {}).get(self.day_key, None)
        msg = f"[{self!s}] Day indices [{self.day_key}] not found. No day split used."
        self.day_idx = self._check_if_idx_none(days, msg, 0, n)

        wear = kwargs.get("wear", None)
        msg = f"[{self!s}] Wear detection not provided. Assuming 100% wear time."
        self.wear_idx = self._check_if_idx_none(wear, msg, 0, n)

    def save_results(self, results, file_name):
        """
        Save the results of the processing pipeline to a csv file

        Parameters
        ----------
        results : dict
            Dictionary of results from the output of predict
        file_name : str
            File name. Can be optionally formatted (see Notes)

        Notes
        -----
        Available format variables available:

        - date: todays date expressed in yyyymmdd format.
        - name: process name.
        - file: file name used in the pipeline, or "" if not found.
        """
        date = dt_date.today().strftime("%Y%m%d")

        file_name = file_name.format(date=date, name=self._name, file=self._file_name)

        DataFrame(results).to_csv(file_name, index=False)

    def _setup_plotting(self, save_name):
        """
        Setup plotting. If this needs to be available to the end user, it should be aliased as
        `setup_plotting` inside __init__

        >>> class NewClass(_BaseProcess)
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.setup_plotting = self._setup_plotting
        """
        pass
