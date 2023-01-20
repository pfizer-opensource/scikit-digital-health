"""
CSV reading process

Lukas Adamowicz
Copyright (c) 2023. Pfizer Inc. All rights reserved
"""
from warnings import warn

from numpy import tile, arange, mean, diff, asarray, argmin, abs, vstack, unique, all as npall, int_
from pandas import read_csv, to_datetime, to_timedelta, Timedelta

from skdh.base import BaseProcess
from skdh.io.base import check_input_file


def handle_timestamp_inconsistency(df, fill_gaps, accel_col_names, accel_in_g, g):
    """
    Handle any time gaps, or timestamps that are only down to the second.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing timestamps and acceleration data.
    fill_gaps : bool
        Fill data gaps with the vector [0, 0, 1] or [0, 0, `g`] depending on
        accel units.
    accel_col_names : array-like
        Array-like (size 3) of the acceleration column names in XYZ order.
    accel_in_g : bool
        If acceleration values are already in units of "g".
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with corrected timestamps.
    fs : float
        Number of samples per second.
    """
    # get a sampling rate. If non-unique timestamps, this will be updated
    n_samples = mean(1 / diff(df['_datetime_'][:2500]).astype(int)) * 1e9  # datetime diff is in ns

    # first check if we have non-unique timestamps
    nonuniq_ts = df["_datetime_"].iloc[1] == df["_datetime_"].iloc[0]

    # if there are non-unique timestamps, fix
    if nonuniq_ts:
        # check that all the blocks are the same size (or that there is only 1 non-equal block
        # at the end)
        _, counts = unique(df["_datetime_"], return_counts=True)
        if not npall(counts[:-1] == counts[0]):
            raise ValueError(
                "Blocks of non-unique timestamps are not all equal size. "
                "Unable to continue reading data."
            )
        # check if the last block is the same size
        if counts[-1] != counts[0]:
            # drop the last blocks worth of data
            warn("Non integer number of blocks. Trimming partial block.", UserWarning)
            df = df.iloc[0:-counts[-1]]

        # get the number of samples, and the number of blocks
        n_samples = counts[0]
        n_blocks = df.shape[0] / n_samples

        # compute time delta to add
        t_delta = tile(arange(0, 1, 1 / n_samples), int(n_blocks))
        t_delta = to_timedelta(t_delta, unit='s')

        # add the time delta so that we have unique timestamps
        df['_datetime_'] += t_delta

    # check if we are filling gaps or not
    if fill_gaps:
        # now fix any data gaps: set the index as the datetime, and then upsample to match
        # the sampling frequency. This will put nan values in any data gaps
        df_full = df.set_index('_datetime_').asfreq(f'{1 / n_samples}S')

        # put the datetime array back in the dataframe
        df_full = df_full.reset_index(drop=False)

        z_fill = 1.0 if accel_in_g else g

        df_full[accel_col_names[0]].fillna(value=0.0, inplace=True)
        df_full[accel_col_names[1]].fillna(value=0.0, inplace=True)
        df_full[accel_col_names[2]].fillna(value=z_fill, inplace=True)
    else:
        # if not filling data gaps, check that there are not gaps that would cause
        # garbage outputs from downstream algorithms
        time_deltas = diff(df['_datetime_']).astype(int) / 1e9  # convert to seconds
        if (abs(time_deltas) > (1.5 / n_samples)).any():
            raise ValueError("There are data gaps in the data, which could potentially result in garbage outputs from downstream algorithms.")

        df_full = df.copy()

    return df_full, float(n_samples)


def handle_windows(time_dt, bases, periods, run_windowing):
    """
    Handle computation of the indices for day windows.

    Parameters
    ----------
    time_dt : pandas.Series
        Timestamp array of pandas datetimes.
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
    if not run_windowing:
        return {}

    start_date = time_dt.iloc[0]
    end_date = time_dt.iloc[-1]

    days = {}
    day_dt = Timedelta(1, unit='day')

    for base, period in zip(bases, periods):
        starts, stops = [], []

        period2 = (base + period) % 24

        t_base = start_date.replace(hour=base, minute=0, second=0, microsecond=0) - day_dt
        t_period = start_date.replace(hour=period2, minute=0, second=0, microsecond=0) - day_dt

        if t_period <= t_base:
            t_period += day_dt
        while t_period < start_date:  # make sure at least one of the indices is during recording
            t_base += day_dt
            t_period += day_dt

        # iterate over the times
        while t_base < end_date:
            starts.append(argmin(abs(time_dt - t_base)))
            stops.append(argmin(abs(time_dt - t_period)))

            t_base += day_dt
            t_period += day_dt

        days[(base, period)] = vstack((starts, stops)).T

    return days


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


class ReadCSV(BaseProcess):
    """
    Read a comma-separated value (CSV) file into memory.

    Parameters
    ----------
    time_col_name : str
        The name of the column containing timestamps.
    accel_col_names : list-like
        List-like of acceleration column names. Must be length 3, in X, Y, Z axis order.
    fill_gaps : bool, optional
        Fill any gaps in acceleration data with the vector [0, 0, 1]. Default is True. If False
        and data gaps are detected, then the reading will raise a `ValueError`.
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

    Notes
    -----
    In order to handle windowing, data gap filling, or timestamp interpolation in
    the case that timestamps are only down to the second (ie ActiGraph CSV files),
    the time column is always first converted to a `datetime64` Series via
    :py:class:`pandas.to_datetime`. To make sure this conversion applies correctly,
    specify whatever key-word arguments to `to_datetime_kwargs`. This includes specifying
    the unit (e.g. `s`, `ms`, `us`, `ns`, etc) if a unix timestamp integer is provided.
    """
    def __init__(
            self,
            time_col_name,
            accel_col_names,
            fill_gaps=True,
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
            fill_gaps=fill_gaps,
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
        self.fill_gaps = fill_gaps
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

        # convert time column to a datetime column. Give a unique name so we shouldnt overwrite
        raw["_datetime_"] = to_datetime(raw[self.time_col_name], **self.to_datetime_kw)

        # now handle data gaps and second level timestamps, etc
        raw, fs = handle_timestamp_inconsistency(raw, self.fill_gaps, self.acc_col_names, self.accel_in_g, self.g_value)

        # first do the windowing
        day_windows = handle_windows(raw["_datetime_"], self.bases, self.periods, self.window)

        # get the time values and convert to seconds
        time = raw["_datetime_"].astype(int).values / 1e9  # int gives ns, convert to s

        # get the acceleration values and convert if necessary
        accel = handle_accel(raw, self.acc_col_names, self.accel_in_g, self.g_value)

        kwargs.update(
            {
                "file": file,
                self._time: time,
                self._acc: accel,
                self._days: day_windows,
                "fs": fs,
            }
        )

        return (kwargs, None) if self._in_pipeline else kwargs
