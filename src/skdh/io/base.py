"""
Base process for file IO

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from pathlib import Path
import functools
from warnings import warn

from numpy import nonzero
from pandas import to_datetime

from skdh.base import BaseProcess
from skdh.utility.exceptions import FileSizeError


def check_input_file(
    extension,
    check_size=True,
    ext_message="File extension [{}] does not match expected [{}]",
):
    """
    Check the input file for existence and suffix.

    Parameters
    ----------
    extension : str
        Expected file suffix, eg '.abc'.
    check_size : bool, optional
        Check file size is over 1kb. Default is True.
    ext_message : str, optional
        Message to print if the suffix does not match. Should take 2 format arguments
        ('{}'), the first for the actual file suffix, and the second for the
        expected suffix.
    """

    def decorator_check_input_file(func):
        @functools.wraps(func)
        def wrapper_check_input_file(self, **kwargs):
            file = kwargs.get("file")
            # check if the file is provided
            if file is None:
                raise ValueError("`file` must not be None.")

            # make a path instance for ease of use
            pfile = Path(file)
            # make sure the file exists
            if not pfile.exists():
                raise FileNotFoundError(f"File {file} does not exist.")

            # check that the file matches the expected extension
            if pfile.suffix != extension:
                if self.ext_error == "warn":
                    warn(ext_message.format(pfile.suffix, extension), UserWarning)
                elif self.ext_error == "raise":
                    raise ValueError(ext_message.format(pfile.suffix, extension))
                elif self.ext_error == "skip":
                    return kwargs

            # check file size if desired
            if pfile.stat().st_size < 1000:
                if check_size and not hasattr(self, "_skip_file_size_check"):
                    raise FileSizeError("File is less than 1kb, nothing to read.")
                elif hasattr(self, "_skip_file_size_check"):
                    warn(
                        f"File is less than 1kb, but the file size check has been "
                        f"skipped, returning empty dictionary: {file}",
                        UserWarning,
                    )
                    return {}

            return func(self, **kwargs)

        return wrapper_check_input_file

    return decorator_check_input_file


def handle_naive_timestamps(time, is_local, tz_name=None):
    """
    Check timestamps to make sure they are either naive, or UTC with a time-zone
    name available.

    Parameters
    ----------
    time : numpy.ndarray
        Array of timestamps.
    is_local : bool
        If the timestamps are naive and represent local time.
    tz_name : {None, str}
        IANA time-zone name.
    """
    if is_local:
        if tz_name is not None:
            # get offset for the timestamp array based on the first timestamp. This works
            # since naive timestamps don't account for DST changes, and therefore will
            # not have duplicated timestamps (just like UTC).

            # invert since we are going from local to UTC.
            offset = (
                -to_datetime(time[0], unit="s")
                .tz_localize(tz_name)
                .utcoffset()
                .total_seconds()
            )

            time += offset
        else:  # is_local, tz_name is None
            warn(
                "Timestamps are local but naive, and no time-zone information is available. "
                "This may mean that if a DST change occurs during the recording period, "
                "the times will be offset by an hour"
            )
    else:  # not is_local
        if tz_name is None:
            warn(
                "Timestamps are not localized, but no time-zone information is available. "
                "This will cause issues for day-windowing, where the day does not actually "
                "start at midnight due to UTC offset.",
                UserWarning,
            )
    return time


class BaseIO(BaseProcess):
    @staticmethod
    def _to_timestamp(time_str, tz_name):
        ts = to_datetime(time_str)
        if tz_name is not None:
            ts = ts.tz_localize(tz_name)

        return ts.timestamp()

    def trim_data(self, start_key, end_key, tz_name, predict_kw, *, time, **data):
        """
        Trim raw data based on provided date-times

        Parameters
        ----------
        start_key : {None, str}
            Start key for the provided trim start time.
        end_key : {None, str}
            End key for the provided trim end time.
        tz_name : {None, str}
            IANA time-zone name for the recording location. If not provided,
            both the `time` array and provided trim times will be assumed to be naive
            and in the same time-zone.
        predict_kw : dict
            Key-word arguments passed to the predict method.
        """
        trim_start = predict_kw.get(start_key)
        trim_end = predict_kw.get(end_key)

        if trim_start is not None and trim_start is None:
            raise ValueError(
                f"`{start_key=}` was provided but not found in `predict` arguments"
            )
        if trim_end is not None and trim_end is None:
            raise ValueError(
                f"`{end_key=}` was provided but not found in `predict` arguments"
            )

        ts_trim_start = (
            time[0] + 0.001
            if trim_start is None
            else self._to_timestamp(trim_start, tz_name)
        )
        ts_trim_end = (
            time[-1] - 0.001
            if trim_end is None
            else self._to_timestamp(trim_end, tz_name)
        )

        # write it so that if there is any time weirdness, get the data from the "middle",
        # ie the last index before the start time, and the first index after the end time.
        i1 = nonzero(time <= max(time[0], ts_trim_start))[0][-1]
        i2 = nonzero(time >= min(time[-1], ts_trim_end))[0][0]

        res = {self._time: time[i1:i2]}
        res.update({k: v[i1:i2] for k, v in data.items()})

        return res
