import numpy as np

def convert_sfreq_to_sampling_interval(x):
    """
    Convert sfreq in Hz (samples/second) to sampling interval (timedelta64[ns]).
    :param x: np.array | float | int, sampling frequency in samples/second (Hz).
    :return: timedelta64[ns], sampling interval as a time delta.
    """
    return np.timedelta64(1, 'ns') * (1 / x * 10 ** 9)


def from_unix(ts, time_unit='s', utc_offset=0):
    """
    Utility function to convert a unix time to timestamp.

    Parameters
    ----------
    ts : unix time in seconds or series of unix times
    time_unit : 's' ! 'ms' Str, 's' if ts is ~10 ** 9, 'ms' if ts ~10 ** 12
    utc_offset : utc_offset due time zone in hours

    Return
    ------
    float

    """
    time_start = np.datetime64("1970-01-01T00:00:00")
    return ts * np.timedelta64(1, time_unit) + time_start + np.timedelta64(utc_offset * 60 ** 2, "s")


def to_unix(ts, time_unit='s', utc_offset=0):
    """
    Utility function to convert a timestamp to unix time.

    Parameters
    ----------
    ts : timestamp or series of timestamps
    time_unit : 's' ! 'ms' Str, 's' if ts is ~10 ** 9, 'ms' if ts ~10 ** 12

    Return
    ------
    float

    """
    time_start = np.datetime64("1970-01-01T00:00:00")
    return ((ts - time_start) - np.timedelta64(utc_offset, 'h')) / np.timedelta64(1, time_unit)




