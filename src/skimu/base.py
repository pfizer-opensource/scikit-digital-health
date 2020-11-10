"""
Base classes, functions, etc for the skimu library

Lukas Adamowicz
Pfizer DMTI 2020
"""
from datetime import date as dt_date
import logging

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

    def __init__(self, return_result=True, **kwargs):
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
        self._return_result = return_result

        self._kw = kwargs

        self.logger = None
        self.log_filename = None

    def predict(self, *args, **kwargs):
        result = self._predict(*args, **kwargs)

        if self._return_result:
            return result[1]
        else:
            return result[0]

    def enable_logging(self, logger=None, filename='pipeline_log.log', name=None):
        """
        Enable logging during the execution of the pipeline. This will initialize loggers
        for steps already added, and any new steps that are added to the pipeline.

        Parameters
        ----------
        logger : logging.Logger, optional
            Custom logger for logging. If None (default), a logger will be created. See Notes for
            a more detailed description of the logger that will be created.
        filename : str, optional
            Filename/path for the logger. Default is 'pipeline_log.log'.
        name : str, optional
            Name for the logger. Default is None, which will use __name__ (skimu.pipeline)

        Notes
        -----
        If a logger is being created, it is created with the following attributes/parameters:

        - Name is set to __name__ (`name=None`) or `name`
        - Level is set to `INFO`
        - Handler is a `logging.FileHandler` with `filename` for the file
        """
        if logger is None:
            self.log_filename = filename  # save the file name
            self.logger = logging.getLogger(name if name is not None else __name__)
            self.logger.setLevel('INFO')  # set the level for the logger
            self.logger.addHandler(logging.FileHandler(filename=filename))
        else:
            if isinstance(logger, logging.Logger):
                self.logger = logger

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

        - date: todays date expressed in yyyymmdd format
        - name: process name
        """
        date = dt_date.today().strftime('%Y%m%d')
        name = self._name

        file_name = file_name.format(date=date, name=name)

        DataFrame(results).to_csv(file_name, index=False)
