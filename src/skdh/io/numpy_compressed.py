"""
Read from a numpy compressed file.

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import load

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

    def __init__(self, ext_error="warn"):
        super(ReadNumpyFile, self).__init__(ext_error=ext_error)

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

        kwargs.update(
            {self._time: data["time"], self._acc: data["accel"], "file": file}
        )
        if "fs" in data:
            kwargs["fs"] = data["fs"][()]

        return (kwargs, None) if self._in_pipeline else kwargs
