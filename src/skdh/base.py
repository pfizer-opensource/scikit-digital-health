"""
Base classes, functions, etc for the skdh library

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from datetime import date as dt_date
import logging
from pathlib import Path
import functools

from pandas import DataFrame, to_datetime
from numpy import array


def handle_process_returns(*, results_to_kwargs):
    """
    Handle updating inputs (kwargs) with return values as appropriate for the
    `process` methods of `BaseProcess` classes.

    Parameters
    ----------
    results_to_kwargs : bool
        Put the first return (results) into the input kwargs.
    """

    def internal_handler(method):
        @functools.wraps(method)
        def magic(self, **kwargs):
            return_vals = method(self, **kwargs)

            # need an extra layer here due to how *return values handle returns
            # of single dictionaries
            if isinstance(return_vals, dict):
                res = return_vals
                updates = []
            else:
                res, *updates = return_vals

            # warnings for updates length
            if len(updates) > 1:
                raise ValueError(
                    "Too many values to update input with, updates should be a dictionary"
                )

            # if we want to update the input with the results
            if results_to_kwargs:
                try:
                    kwargs.update(res)
                except TypeError as e:
                    raise TypeError(
                        "Cannot update input  with non-dictionary output"
                    ) from e

            # We have updates
            if updates:  # equivalent to len(updates) > 0 or updates != []
                try:
                    kwargs.update(updates[0])
                except TypeError as e:
                    raise TypeError(
                        "Cannot update input with non-dictionary output"
                    ) from e

            # return based on pipeline
            if self._in_pipeline:
                return kwargs, res
            else:
                return res

        return magic

    return internal_handler


class BaseProcess:
    """
    The base class for any Process that is designed to work within the
    Scikit-Digital-Health framework, and the Pipeline class. Should be subclassed.
    """

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
        return self._cls_name

    def __repr__(self):
        ret = f"{self._cls_name}("
        for k in self._kw:
            ret += f"{k}={self._kw[k]!r}, "
        if ret[-1] != "(":
            ret = ret[:-2]
        ret += ")"

        return ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._kw == other._kw
        else:
            return False

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
        self._cls_name = self.__class__.__name__
        self._name = self._cls_name  # overwritten by pipeline as needed
        self._in_pipeline = False  # initialize to false.  Will be set by the pipeline
        self.pipe_save_file = None  # initialize to None, will be set/used by pipeline
        self.pip_plot_file = None  # will be set/used by Pipeline only

        self._kw = kwargs

        self.logger = logging.getLogger(__name__)

        # default day key
        self.day_key = (-1, -1)

        # time zone name
        self.tz_name = None

        # day and wear indices
        self.day_idx = (None, None)
        self.wear_idx = (None, None)

        # file name saving
        self._file_name = ""

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

    def predict(self, expect_days, expect_wear, *args, **kwargs):
        """
        Intended to be overwritten in the subclass. Should still be called
        with super.
        """
        self.logger.info(f"Entering {self._cls_name} processing with call {self!r}")
        # save the filename for saving reference
        self._file_name = Path(kwargs.get("file", "")).stem

        # get the time zone name if available
        self.tz_name = kwargs.get("tz_name", None)

        if expect_days:
            n = kwargs.get(self._acc, kwargs.get(self._time)).shape[0] - 1

            days = kwargs.get(self._days, {}).get(self.day_key, None)
            msg = (
                f"[{self!s}] Day indices [{self.day_key}] not found. No day split used."
            )
            self.day_idx = self._check_if_idx_none(days, msg, 0, n)

        if expect_wear:
            n = kwargs.get(self._acc, kwargs.get(self._time)).shape[0] - 1

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
        - file: file name used in the pipeline, or "" if not found.
        - version: SKDH version number (short form, no period separation)
        """
        # avoid circular import
        from skdh import __skdh_version__ as skdh_version

        date = dt_date.today().strftime("%Y%m%d")
        version = skdh_version.replace(".", "")

        file_name = file_name.format(date=date, file=self._file_name, version=version)

        kw_line = [f"{k}: {self._kw[k]}".replace(",", "  ") for k in self._kw]

        # get the information to save
        lines = [
            "Scikit-Digital-Health\n",
            f"Version,{skdh_version}\n",
            f"Date,{date}\n",
            ",".join(kw_line),
            "\n",
            "\n",
        ]

        with open(file_name, "w") as f:
            f.writelines(lines)

        DataFrame(results).to_csv(file_name, index=False, mode="a")

        return file_name

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

    # TIME helpers
    def convert_timestamps(self, t):
        """
        Convert a timestamp/array of timestamps to a datetime object

        Parameters
        ----------
        t : float, pd.Series, numpy.ndarray
            Unix timestamp in seconds

        Returns
        -------
        datetime.datetime
            Datetime object
        """
        # set if we want to return aware time
        kw = {"unit": "s", "utc": self.tz_name is not None}
        dt = to_datetime(t, **kw)
        if self.tz_name is not None:
            dt = dt.tz_convert(self.tz_name)
        return dt
