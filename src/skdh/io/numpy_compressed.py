"""
Read from a numpy compressed file.

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import load
import pandas as pd
import numpy as np

from skdh.base import BaseProcess
from skdh.io.base import check_input_file


class ReadNumpyFile(BaseProcess):
    """
    Read a Numpy compressed file into memory. The file should have been
    created by `numpy.savez`. The data contained is read in
    unprocessed - ie acceleration is already assumed to be in units of
    'g' and time in units of seconds. No day windowing is performed. Expected
    keys are `time` and `accel`. If `fs` is present, it is used as well.

    Parameters
    ----------
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.npz).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.
    """

    def __init__(self, ext_error="warn", bases=None, periods=None):
        super(ReadNumpyFile, self).__init__(ext_error=ext_error)
        self.bases = list(bases)
        self.periods = list(periods)

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

    @check_input_file(".npz", check_size=True)
    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from a numpy compressed file.

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be
            converted by `str(file)`.

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
        - `fs`: sampling frequency in Hz.
        """
        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        data = load(file)

        # day/windowing stuff - ADDED
        time = pd.to_datetime(data["time"])
        start_date = time[0]
        end_date = time[-1]
        days = {}
        day_dt = pd.Timedelta(1, unit='day')
        for b, p in zip(self.bases, self.periods):
            starts, stops = [], []

            p2 = (b + p) % 24

            tb = start_date.replace(hour=b, minute=0, second=0) - day_dt
            tp = start_date.replace(hour=p2, minute=0, second=0) - day_dt
            if tp <= tb:
                tp += day_dt
            while tp < start_date:  # make sure at least one of the indices is during recording
                tb += day_dt
                tp += day_dt

            # iterate over the times
            while tb < end_date:
                starts.append(np.argmin(abs(time - tb)))
                stops.append(np.argmin(abs(time - tp)))

                tb += day_dt
                tp += day_dt

            days[(b, p)] = np.vstack((starts, stops)).T

        kwargs.update(
            {self._time: data["time"], self._acc: data["accel"], self._temp: data['temperature'], "file": file, self._days: days}
        )
        if "fs" in data:
            kwargs["fs"] = data["fs"][()]

        return (kwargs, None) if self._in_pipeline else kwargs
