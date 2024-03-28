"""
CSV reading process

Lukas Adamowicz
Copyright (c) 2023. Pfizer Inc. All rights reserved
"""
from warnings import warn

from numpy import (
    tile,
    arange,
    mean,
    diff,
    round,
    abs,
    unique,
    all as npall,
    int_,
)
from pandas import read_csv, to_datetime, to_timedelta, date_range

from skdh.base import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file


def _as_list(a):
    if isinstance(a, str):
        return [a]
    elif isinstance(a, list):
        return a
    else:
        return list(a)


def handle_timestamp_inconsistency(df, fill_gaps, column_names, fill_dict):
    """
    Handle any time gaps, or timestamps that are only down to the second.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing timestamps and acceleration data.
    fill_gaps : bool
        Fill data gaps with the vector [0, 0, 1] or [0, 0, `g`] depending on
        accel units.
    column_names : dict
        Dictionary of column names.
    fill_dict : dict
        Dictionary of fill values for columns identified in `column_names`. In cases
        where there are multiple columns for a datastream, the last will be filled with
        the value.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with corrected timestamps.
    fs : float
        Number of samples per second.
    """
    # get a sampling rate. If non-unique timestamps, this will be updated
    n_samples = round(mean(1 / diff(df["_datetime_"][:2500]).astype(int)) * 1e9, decimals=6)
    # datetime diff is in ns

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
            df = df.iloc[0 : -counts[-1]]

        # get the number of samples, and the number of blocks
        n_samples = counts[0]
        n_blocks = df.shape[0] / n_samples

        # compute time delta to add
        t_delta = tile(arange(0, 1, 1 / n_samples), int(n_blocks))
        t_delta = to_timedelta(t_delta, unit="s")

        # add the time delta so that we have unique timestamps
        df.loc[:, "_datetime_"] += t_delta

    # check if we are filling gaps or not
    if fill_gaps:
        # now fix any data gaps: set the index as the datetime, and then upsample to match
        # the sampling frequency. This will put nan values in any data gaps.
        # have to use ms to handle fractional seconds.
        # do this in a round-about way so that we can use `reindex` and specify a tolerance
        t0 = df['_datetime_'].iloc[0]
        t1 = df['_datetime_'].iloc[-1]
        dr = date_range(t0, t1, freq=f"{round(1000 / n_samples, decimals=6)}ms", inclusive='both', name='_datetime_')
        df_full = df.set_index("_datetime_").reindex(index=dr, method='nearest', limit=1, tolerance=to_timedelta(0.1 / n_samples, unit='s'))
        df_full = df_full.reset_index()

        # put the datetime array back in the dataframe
        df_full = df_full.reset_index(drop=False)

        for dstream, stream_cols in column_names.items():
            stream_cols = _as_list(stream_cols)
            try:
                for col in stream_cols[:-1]:
                    df_full[col] = df_full[col].fillna(value=0.0)
                df_full[stream_cols[-1]] = df_full[stream_cols[-1]].fillna(fill_dict.get(dstream, 0.0))
            except KeyError:
                warn(f"Column {col} not found to fill.", UserWarning)
                continue
    else:
        # if not filling data gaps, check that there are not gaps that would cause
        # garbage outputs from downstream algorithms
        time_deltas = diff(df["_datetime_"]).astype(int) / 1e9  # convert to seconds
        if (abs(time_deltas) > (1.5 / n_samples)).any():
            raise ValueError(
                "There are data gaps in the data, which could potentially result in garbage outputs from downstream algorithms."
            )

        df_full = df.copy()

    return df_full, float(n_samples)


class ReadCSV(BaseProcess):
    """
    Read a comma-separated value (CSV) file into memory.

    Parameters
    ----------
    time_col_name : str
        The name of the column containing timestamps.
    column_names : dict
        Dictionary of column names for different data types. See Notes.
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
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.bin).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.
    
    .. deprecated:: 0.14.0
        `bases` Removed in favor of having windowing be its own class,
        :class:`skdh.preprocessing.GetDayWindowIndices`.
        `periods` Removed in favor of having windowing be its own class.

    Notes
    -----
    For `column_names`, valid keys are:

    - accel
    - gyro
    - ecg
    - temperature

    For a key, either strings or lists of strings are accepted. If multiple columns
    are provided for different axes, they are assumed to be in X, Y, Z order.

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
        column_names,
        fill_gaps=True,
        to_datetime_kwargs=None,
        accel_in_g=True,
        g_value=9.81,
        read_csv_kwargs=None,
        ext_error="warn",
    ):
        if to_datetime_kwargs is None:
            to_datetime_kwargs = {}
        if read_csv_kwargs is None:
            read_csv_kwargs = {}

        super().__init__(
            time_col_name=time_col_name,
            column_names=column_names,
            fill_gaps=fill_gaps,
            to_datetime_kwargs=to_datetime_kwargs,
            accel_in_g=accel_in_g,
            g_value=g_value,
            read_csv_kwargs=read_csv_kwargs,
            ext_error=ext_error,
        )

        self.time_col_name = time_col_name
        self.column_names = column_names
        self.fill_gaps = fill_gaps
        self.to_datetime_kw = to_datetime_kwargs
        self.accel_in_g = accel_in_g
        self.g_value = g_value
        self.read_csv_kwargs = read_csv_kwargs

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(".csv")
    def predict(self, *, file, tz_name=None, **kwargs):
        """
        predict(*, file)

        Read the data from a comma-separated value (CSV) file.

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read.
        tz_name : {None, str}, optional
            Name of a time-zone to convert the timestamps to. Default is None,
            which will leave them as naive.

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
        fill_values = {
            "accel": 1.0 if self.accel_in_g else self.g_value,
            "gyro": 0.0,
            "ecg": 0.0,
            "temperature": 0.0,
        }

        super().predict(expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs)

        # load the file with pandas
        raw = read_csv(file, **self.read_csv_kwargs)

        # update the to_datetime_kwargs based on tz_name.  tz_name==None (utc=False)
        self.to_datetime_kw.update({"utc": tz_name is not None})

        # convert time column to a datetime column. Give a unique name so we shouldnt overwrite
        raw["_datetime_"] = to_datetime(raw[self.time_col_name], **self.to_datetime_kw)

        # convert timestamps if necessary
        if tz_name is not None:
            # convert, and then remove the timezone so its naive again, but now in local time
            tmp = raw["_datetime_"].dt.tz_convert(tz_name)
            raw["_datetime_"] = tmp.dt.tz_localize(None)

        # now handle data gaps and second level timestamps, etc
        raw, fs = handle_timestamp_inconsistency(
            raw, self.fill_gaps, self.column_names, fill_values
        )

        # get the time values and convert to seconds
        time = raw["_datetime_"].astype(int).values / 1e9  # int gives ns, convert to s

        # setup the results dictionary
        results = {
            self._time: time,
            "fs": fs,
        }

        # grab the data we expect
        for dstream in self.column_names:
            try:
                results[dstream] = raw[self.column_names[dstream]].values
            except KeyError:
                warn(f"Data stream {dstream} specified in column names but all columns {self.column_names[dstream]} not found in the read data. Skipping.")
                continue
        
        # convert accel data
        if 'accel' in results and not self.accel_in_g:
            results['accel'] /= self.g_value

        return results
