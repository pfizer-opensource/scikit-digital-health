"""
GeneActiv reading process

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from skdh.base import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file, handle_naive_timestamps
from skdh.io._extensions import read_geneactiv


class ReadBin(BaseProcess):
    """
    Read a binary .bin file from a GeneActiv sensor into memory. Acceleration values are returned
    in units of `g`. If providing a base and period value, included in the output will be the
    indices to create windows starting at the `base` time, with a length of `period`.

    Parameters
    ----------
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.bin).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.

    Examples
    ========
    Setup a reader:

    >>> reader = ReadBin()
    >>> reader.predict('example.bin')
    {'accel': ..., 'time': ...}
    """

    def __init__(self, ext_error="warn"):
        super().__init__(
            # kwargs
            ext_error=ext_error,
        )

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(".bin")
    def predict(self, *, file, tz_name=None, **kwargs):
        """
        predict(*, file)

        Read the data from the GeneActiv file

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be converted by
            `str(file)`.
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
            If the file name is not provided

        Notes
        -----
        The keys in `data` depend on which data the file contained. Potential keys are:

        - `accel`: acceleration [g]
        - `time`: timestamps [s]
        - `light`: light values [unknown]
        - `temperature`: temperature [deg C]
        """
        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs
        )

        # read the file
        n_max, fs, acc, time, light, temp = read_geneactiv(str(file))

        results = {
            self._time: handle_naive_timestamps(
                time[:n_max], is_local=True, tz_name=tz_name
            ),
            self._acc: acc[:n_max, :],
            self._temp: temp[:n_max],
            "light": light[:n_max],
            "fs": fs,
        }

        return results
