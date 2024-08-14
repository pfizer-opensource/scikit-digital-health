"""
Read from a numpy compressed file.

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from warnings import warn

from numpy import load as np_load

from skdh.base import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file, handle_naive_timestamps


class ReadNumpyFile(BaseProcess):
    """
    Read a Numpy compressed file into memory. The file should have been
    created by `numpy.savez`. The data contained is read in
    unprocessed - ie acceleration is already assumed to be in units of
    'g' and time in units of seconds. No day windowing is performed. Expected
    keys are `time` and `accel`. If `fs` is present, it is used as well.

    Parameters
    ----------
    time_is_local : bool, optional
        If any timestamp arrays are naive/local. Default is True. If True, and a `tz_name`
        is provided, output timestamps will be UTC, and require conversion to the
        local time.
    allow_pickle : bool, optional
        Allow pickled objects in the NumPy file. Default is False, which is the safer option.
        For more information see :py:meth:`numpy.load`.
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.npz).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.
    """

    def __init__(self, time_is_local=True, allow_pickle=False, ext_error="warn"):
        super(ReadNumpyFile, self).__init__(
            time_is_local=time_is_local, allow_pickle=allow_pickle, ext_error=ext_error
        )

        self.time_is_local = time_is_local

        self.allow_pickle = allow_pickle

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(".npz", check_size=True)
    def predict(self, *, file, tz_name=None, **kwargs):
        """
        predict(*, file, tz_name=None)

        Read the data from a numpy compressed file.

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be
            converted by `str(file)`.
        tz_name : {None, str}, optional
            IANA time-zone name for the recording location. If not provided, timestamps
            will represent local time naively. This means they will not account for
            any time changes due to Daylight Saving Time.

        Returns
        -------
        data : dict
            Dictionary of the data contained in the file.

        Raises
        ------
        ValueError
            If the file name is not provided.

        Notes
        -----
        The keys in `data` depend on which data the file contained. Potential keys are:

        - `accel`: acceleration [g]
        - `time`: timestamps [s]
        - `fs`: sampling frequency in Hz.
        """
        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs
        )

        # warn if time is not local and we dont have a timezone
        if not self.time_is_local and tz_name is None:
            warn(
                "Time is not local, but no timezone is provided. "
                "Output timestamps will be in UTC."
            )

        results = {}

        with np_load(file, allow_pickle=self.allow_pickle) as data:
            results.update(data)  # pull everything in
            # make sure that fs is saved properly
            if "fs" in data:
                results["fs"] = data["fs"][()]

        # check that time and accel are in the correct names
        if self._time not in results or self._acc not in results:
            raise ValueError(
                f"Missing `{self._time}` or `{self._acc}` arrays in the file"
            )

        # handle the timestamps
        results[self._time] = handle_naive_timestamps(
            results[self._time], is_local=self.time_is_local, tz_name=tz_name
        )

        return results
