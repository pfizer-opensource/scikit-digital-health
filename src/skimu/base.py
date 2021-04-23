"""
Base classes, functions, etc for the skimu library

Lukas Adamowicz
Pfizer DMTI 2020
"""
from datetime import date as dt_date
import logging
from pathlib import Path

from pandas import DataFrame


class _BaseProcess:
    # names of the variables that are passed to predict
    # CHANGE IF predict/_predict call changes!
    _file = 'file'
    _time = 'time'
    _acc = 'accel'
    _gyro = 'gyro'
    _mag = 'magnet'
    _temp = 'temperature'
    _days = 'day_ends'

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

        # for plotting
        self.f = self.ax = self.plot_fname = None

    def predict(self, *args, **kwargs):
        """
        Intended to be overwritten in the subclass. Should still be called with super though
        """
        self.logger.info(f"Entering {self._name} processing with call {self!r}")
        # save the filename for saving reference
        fname = kwargs.get("file", "")
        self._file_name = Path(fname if fname is not None else "").stem

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
        date = dt_date.today().strftime('%Y%m%d')

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
