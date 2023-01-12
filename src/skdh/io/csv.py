"""
CSV reading process

Lukas Adamowicz
Copyright (c) 2023. Pfizer Inc. All rights reserved
"""
from warnings import warn

from numpy import tile, arange, mean, diff, asarray, int_
from pandas import read_csv, to_datetime

from skdh.base import BaseProcess
from skdh.io.base import check_input_file


def handle_timestamps(time_series, is_seconds, to_datetime_kw):
    """
    Handle converting time to unix seconds.

    Parameters
    ----------
    time_series : pandas.Series
        Series of timestamps.
    is_seconds : bool
        If the time values are already in unix seconds.
    to_datetime_kw : dict
        Key-word arguments to pass to :py:class:`pandas.to_datetime`.

    Returns
    -------
    time : numpy.ndarray
        Array of epoch timestamps in seconds.
    fs : float
        Sampling frequency
    """
    if is_seconds:
        time = time_series.values
    else:
        # convert to datetime
        time_dt = to_datetime(time_series, **to_datetime_kw)

        # TODO add timezone conversion?
        # change to epoch timestamps in seconds
        time = time_dt.astype(int).values / 1e9  # int returns nanoseconds

    # calculate the sampling frequency
    fs = mean(1 / diff(time[:2500]))
    # check if we have duplicated timestamps (ie 1 time value for multiple samples)
    # this is common with actigraph timestamps
    n_samples = (time[:2500] == time[0]).sum()
    if n_samples != 1:
        # set fs
        fs = float(n_samples)
        # check that there are an even number of blocks
        n_blocks = time.size / n_samples

        if int(n_samples) != n_samples:
            warn("Non integer number of blocks. Trimming partial block.", UserWarning)
            N = int(time.size // n_samples * n_samples)
            # drop in-place
            time = time[:N]

        t_delta = tile(arange(0, 1, 1 / n_samples), int(n_blocks))

        time += t_delta

    return time


def handle_accel(df, accel_cols, acc_in_g, g):
    """
    Extract the acceleration columns from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas dataframe containing data from the CSV file.
    accel_cols : list-like
        List of acceleration column names in X/Y/Z axis order.
    acc_in_g : bool
        If the data is already in units of "g".
    g : float
        Gravitational acceleration.

    Returns
    -------
    accel : numpy.ndarray
        (N, 3) array of acceleration values in units of "g".
    """
    acc = df[accel_cols].values

    if not acc_in_g:
        acc /= g

    return acc


def handle_windows(time, bases, periods, run_windowing):
    """
    Handle computation of the indices for day windows.

    Parameters
    ----------
    time : numpy.ndarray
        Timestamp array.
    bases : list-like
        List of base times at which windows start. 24hr format.
    periods : list-like
        List of window lengths for each `base` time.
    run_windowing : bool
        Compute day windows.

    Returns
    -------
    day_windows : dict
        Dictionary of numpy arrays containing the indices for the desired days.
    """
    pass


class ReadCSV(BaseProcess):
    """
    Read a comma-separated value (CSV) file into memory.

    Parameters
    ----------
    time_col_name : str
        The name of the column containing timestamps.
    accel_col_names : list-like
        List-like of acceleration column names. Must be length 3, in X, Y, Z axis order.
    time_is_seconds : bool, optional
        Provided time data is already in unix seconds (ie seconds from 1970-01-01).
        If false, will attempt to use :py:class:`pandas.to_datetime` to convert.
        Default is False.
    to_datetime_kwargs : dict, optional
        Dictionary of key-word arguments for :py:class:`pandas.to_datetime`.
    accel_in_g : bool, optional
        If the acceleration values are in units of "g". Default is True.
    g_value : float, optional
        Gravitational acceleration. Default is 9.81 m/s^2.
    read_csv_kwargs : None, dict, optional
        Dictionary of additional key-word arguments for :py:class:`pandas.read_csv`.
    bases : {None, int, list-like}, optional
        Base hours [0, 23] in which to start a window of time. Default is None,
        which will not do any windowing. Both `base` and `period` must be defined
        in order to window. Can use multiple, but the number of `bases` must match
        the number of `periods`.
    periods : {None, int, list-like}, optional
        Periods for each window, in [1, 24]. Defines the number of hours per window.
        Default is None, which will do no windowing. Both `period` and `base` must
        be defined to window. Can use multiple but the number of `periods` must
        match the number of `bases`.
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.bin).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.
    """
    def __init__(
            self,
            time_col_name,
            accel_col_names,
            time_is_seconds=False,
            to_datetime_kwargs=None,
            accel_in_g=True,
            g_value=9.81,
            read_csv_kwargs=None,
            bases=None,
            periods=None,
            ext_error='warn'
    ):
        if to_datetime_kwargs is None:
            to_datetime_kwargs = {}
        if read_csv_kwargs is None:
            read_csv_kwargs = {}

        super().__init__(
            time_col_name=time_col_name,
            accel_col_names=accel_col_names,
            time_is_seconds=time_is_seconds,
            to_datetime_kwargs=to_datetime_kwargs,
            accel_in_g=accel_in_g,
            g_value=g_value,
            read_csv_kwargs=read_csv_kwargs,
            bases=bases,
            periods=periods,
            ext_error=ext_error,
        )

        self.time_col_name = time_col_name
        self.acc_col_names = accel_col_names
        self.time_is_seconds = time_is_seconds
        self.to_datetime_kw = to_datetime_kwargs
        self.accel_in_g = accel_in_g
        self.g_value = g_value
        self.read_csv_kwargs = read_csv_kwargs

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

        if (bases is None) and (periods is None):
            self.window = False
            self.bases = asarray([0])  # needs to be defined for passing to extensions
            self.periods = asarray([12])
        elif (bases is None) or (periods is None):
            warn("One of base or period is None, not windowing", UserWarning)
            self.window = False
            self.bases = asarray([0])
            self.periods = asarray([12])
        else:
            if isinstance(bases, int) and isinstance(periods, int):
                bases = asarray([bases])
                periods = asarray([periods])
            else:
                bases = asarray(bases, dtype=int_)
                periods = asarray(periods, dtype=int_)

            if ((0 <= bases) & (bases <= 23)).all() and (
                (1 <= periods) & (periods <= 24)
            ).all():
                self.window = True
                self.bases = bases
                self.periods = periods
            else:
                raise ValueError(
                    "Base must be in [0, 23] and period must be in [1, 23]"
                )

    @check_input_file(".csv")
    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from a comma-separated value (CSV) file.

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read

        Returns
        -------
        data : dict
            Dictionary of the time and acceleration data contained in the file.
            Time will be in unix seconds, and acceleration will be in units of "g".

        Raises
        ------
        ValueError
            If the file name is not provided.
        """

        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        # load the file with pandas
        raw = read_csv(file, **self.read_csv_kwargs)

        # get the time values and convert if necessary
        time, fs = handle_timestamps(raw[self.time_col_name], self.time_is_seconds, self.to_datetime_kw)

        # get the acceleration values and convert if necessary
        accel = handle_accel(raw, self.acc_col_names, self.accel_in_g, self.g_value)

        # make sure that if we trimmed time, we trim acceleration
        accel = accel[:time.size, :]

        # handle the windowing
        day_windows = handle_windows(time, self.bases, self.periods, self.window)

        kwargs.update(
            {
                "file": file,
                self._time: time,
                self._acc: accel,
                "fs": fs,
            }
        )

        return (kwargs, None) if self._in_pipeline else kwargs
