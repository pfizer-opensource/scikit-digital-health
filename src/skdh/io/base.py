"""
Base process for file IO

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from pathlib import Path
import functools
from warnings import warn

from pandas import to_datetime

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
