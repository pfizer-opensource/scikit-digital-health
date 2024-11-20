"""
GeneActiv reading process

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from skdh.base import handle_process_returns
from skdh.io.base import check_input_file, handle_naive_timestamps, BaseIO
from skdh.io._extensions import read_geneactiv


class ReadBin(BaseIO):
    """
    Read a binary .bin file from a GeneActiv sensor into memory. Acceleration values are returned
    in units of `g`. If providing a base and period value, included in the output will be the
    indices to create windows starting at the `base` time, with a length of `period`.

    Parameters
    ----------
    trim_keys : {None, tuple}, optional
        Trim keys provided in the `predict` method. Default (None) will not do any trimming.
        Trimming of either start or end can be accomplished by providing None in the place
        of the key you do not want to trim. If provided, the tuple should be of the form
        (start_key, end_key). When provided, trim datetimes will be assumed to be in the
        same timezone as the data (ie naive if naive, or in the timezone provided).
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

    def __init__(self, trim_keys=None, ext_error="warn"):
        super().__init__(
            # kwargs
            trim_keys=trim_keys,
            ext_error=ext_error,
        )

        self.trim_keys = trim_keys

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

        # trim the data if necessary
        if self.trim_keys is not None:
            results = self.trim_data(
                *self.trim_keys,
                tz_name,
                kwargs,
                time=handle_naive_timestamps(
                    time[:n_max], is_local=True, tz_name=tz_name
                ),
                **{
                    self._acc: acc[:n_max, :],
                    self._temp: temp[:n_max],
                    "light": light[:n_max],
                },
            )
            results["fs"] = fs
        else:
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
