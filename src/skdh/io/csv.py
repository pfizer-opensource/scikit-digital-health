"""
CSV reading process

Lukas Adamowicz
Copyright (c) 2023. Pfizer Inc. All rights reserved
"""

from warnings import warn

from numpy import (
    array,
    tile,
    arange,
    median,
    diff,
    round,
    abs,
    unique,
    any as npany,
    allclose,
    nonzero,
    diff,
)
from pandas import read_csv, to_datetime

from skdh.base import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file
from skdh.utility.internal import fill_data_gaps


class ReadCSV(BaseProcess):
    """
    Read a comma-separated value (CSV) file into memory.

    Parameters
    ----------
    time_col_name : str
        The name of the column containing timestamps.
    column_names : dict
        Dictionary of column names for different data types. See Notes.
    drop_duplicate_timestamps : bool, optional
        Drop duplicate timestamps before doing any timestamp handling or gap filling.
        Default is False.
    trim : {None, 'start', 'end', 'both'}, optional
        Trim the recording/data length to times provided to the `predict` method. Default
        is None, which will trim nothing. 'start' will trim the start of the recording
        to after the `trim_start` parameter, 'end' will trim the end of the recording
        to be the `trim_end` parameter. Both will trim at both ends. Timestrings 
        will be assumed to be in the same timezone as the data, such that if you pass
        `tz_name` to the predict method, the trim times will be taken as in that timezone.
    fill_gaps : bool, optional
        Fill any gaps in data streams. Default is True. If False and data gaps are
        detected, then the reading will raise a `ValueError`.
    fill_value : {None, dict}, optional
        Dictionary with keys and values to fill data streams with. See Notes for
        default values if not provided.
    gaps_error : {'raise', 'warn', 'ignore'}, optional
        Behavior if there are large gaps in the datastream after handling timestamps.
        Default is to raise an error. NOT recommended to change unless the data
        is being read as part of a :class:`skdh.io.MultiReader` call, in which case
        it will likely be re-sampled.
    to_datetime_kwargs : dict, optional
        Dictionary of key-word arguments for :py:class:`pandas.to_datetime`.
    raw_conversions : dict, optional
        Conversions to apply to raw data, with keys matching those in `column_names`.
        Conversions are applied by dividing the raw data stream by the conversion factor
        provided. If left as None, no conversions will be applied (ie conversion
        factor of 1.0).
    accel_in_g : bool, optional
        If the acceleration values are in units of "g". Default is True.

        .. deprecated:: 0.15.1
            Use `raw_conversions` instead.

    g_value : float, optional
        Gravitational acceleration. Default is 9.81 m/s^2.

        .. deprecated:: 0.15.1
            Use `raw_conversions` instead.

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

    Default fill values are:

    - accel: numpy.array([0.0, 0.0, 1.0])
    - gyro: 0.0
    - temperature: 0.0
    - ecg: 0.0
    """

    def __init__(
        self,
        time_col_name,
        column_names,
        drop_duplicate_timestamps=False,
        trim=None,
        fill_gaps=True,
        fill_value=None,
        gaps_error="raise",
        to_datetime_kwargs=None,
        raw_conversions=None,
        accel_in_g=True,
        g_value=9.81,
        read_csv_kwargs=None,
        ext_error="warn",
    ):
        super().__init__(
            time_col_name=time_col_name,
            column_names=column_names,
            drop_duplicate_timestamps=drop_duplicate_timestamps,
            trim=trim,
            fill_gaps=fill_gaps,
            fill_value=None,
            gaps_error=gaps_error,
            to_datetime_kwargs=to_datetime_kwargs,
            raw_conversions=raw_conversions,
            read_csv_kwargs=read_csv_kwargs,
            ext_error=ext_error,
        )

        if to_datetime_kwargs is None:
            to_datetime_kwargs = {}
        if read_csv_kwargs is None:
            read_csv_kwargs = {}

        # raw conversions
        if raw_conversions is None:
            raw_conversions = {k: 1.0 for k in column_names}
            if not accel_in_g:
                warn(
                    "Parameter accel_in_g is deprecated in favor of raw_conversions",
                    DeprecationWarning,
                )
                raw_conversions["accel"] = g_value

        if gaps_error.lower() not in ["raise", "warn", "ignore"]:
            raise ValueError("gaps_error must be one of `raise`, `warn`, or `ignore`.")

        self.time_col_name = time_col_name
        self.column_names = column_names
        self.trim = trim
        self.fill_gaps = fill_gaps
        self.fill_value = {} if fill_value is None else fill_value
        self.gaps_error = gaps_error
        self.drop_dupl_time = drop_duplicate_timestamps
        self.to_datetime_kw = to_datetime_kwargs
        self.raw_conversions = raw_conversions
        self.read_csv_kwargs = read_csv_kwargs

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

    def handle_gaps_error(self, msg):
        if self.gaps_error == "raise":
            raise ValueError(msg)
        elif self.gaps_error == "warn":
            warn(msg)
        else:
            pass

    def handle_timestamp_inconsistency_np(self, fill_dict, time, data):
        """
        Handle any time gaps, or timestamps that are only down to the second.

        Parameters
        ----------
        fill_dict : dict
            Dictionary of fill values for columns identified in `column_names`. In cases
            where there are multiple columns for a datastream, the last will be filled with
            the value.
        time : numpy.ndarray
            Array of timestamps in unix seconds.
        data : dict
            Dictionary of data streams

        Returns
        -------
        fs : float
            Number of samples per second.
        time : numpy.ndarray
            Timestamp array with update timestamps to be unique and gaps filled
        data : dict
            Data dictionary with updated arrays with gaps filled if specified.
        """
        if self.drop_dupl_time:
            time, unq_ind = unique(time, return_index=True)
            for name, dstream in data.items():
                data[name] = dstream[unq_ind]
        # get a sampling rate. If non-unique timestamps, this will be updated
        n_samples = round(median(1 / diff(time[:2500])), decimals=6)

        # first check if we have non-unique timestamps
        nonuniq_ts = time[1] == time[0]

        # if there are non-unique timestamps, fix
        if nonuniq_ts:
            # check that all the blocks are the same size (or that there is only 1 non-equal block
            # at the end)
            block_changes = nonzero(diff(time, prepend=time[0], append=time[-1] + 1))[
                0
            ]  # get a mask of where blocks change
            counts = diff(block_changes, prepend=0)
            # check if the last block is the same size
            if counts[-1] != counts[0]:
                # drop the last blocks worth of data
                warn(
                    "Non integer number of blocks. Trimming partial block.", UserWarning
                )
                time = time[: -counts[-1]]
                for name, dstream in data.items():
                    data[name] = dstream[: -counts[-1]]

                counts = counts[:-1]  # remove the last block count

            # now check if all remaining blocks are the same size
            if not allclose(counts, counts[0]):
                raise ValueError(
                    "Blocks of non-unique timestamps are not all equal size. "
                    "Unable to continue reading data."
                )

            # get the number of samples, and the number of blocks
            n_samples = counts[0]
            n_blocks = time.size / n_samples

            # compute time delta to add
            t_delta = tile(arange(0, 1, 1 / n_samples), int(n_blocks))

            # add the time delta so that we have unique timestamps
            time += t_delta

        # check if we are filling gaps or not
        if self.fill_gaps:
            time_rs, data = fill_data_gaps(time, n_samples, fill_dict, **data)
        else:
            # if not filling data gaps, check that there are not gaps that would
            # cause garbage outputs from downstream algorithms
            time_deltas = diff(time)
            if npany(abs(time_deltas) > (1.5 / n_samples)):
                self.handle_gaps_error(
                    "There are data gaps, which could potentially results in incorrect outputs from downstream algorithms"
                )

            time_rs = time

        return n_samples, time_rs, data

    def trim_data(self, df, trim_start, trim_end, tz_name):
        """
        Trim data to provided start/end times.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw data
        trim_start : {None, str}
            Trim start time
        trim_end : {None, str}
            Trime end time
        tz_name : {None, str}
            Timezone name for the data

        Returns
        -------
        df_trimmed : pandas.DataFrame
        """
        if self.trim.lower() == "start":
            start = to_datetime(trim_start, utc=False)
            if tz_name is not None:
                start = start.tz_localize(tz_name)
            end = df[self.time_col_name].iloc[-1]
        elif self.trim.lower() == "end":
            end = to_datetime(trim_end, utc=False)
            if tz_name is not None:
                end = end.tz_localize(tz_name)
            start = df[self.time_col_name].iloc[0]
        elif self.trim.lower() == "both":
            start = to_datetime(trim_start, utc=False)
            if tz_name is not None:
                start = start.tz_localize(tz_name)
            end = to_datetime(trim_end, utc=False)
            if tz_name is not None:
                end = end.tz_localize(tz_name)
        else:
            raise ValueError("Invalid trim value, must be one of {'start', 'end', 'both'}")

        i1 = nonzero(df[self.time_col_name] <= start)[0][-1]
        i2 = nonzero(df[self.time_col_name] >= end)[0][0]
        
        return df.iloc[i1:i2]

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
            IANA time-zone name for the recording location. If not provided, timestamps
            will represent local time naively. This means they will not account for
            any time changes due to Daylight Saving Time.

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
            "accel": self.fill_value.get(
                "accel", array([0.0, 0.0, self.raw_conversions.get("accel", 1.0)])
            ),
            "gyro": self.fill_value.get("gyro", 0.0),
            "ecg": self.fill_value.get("ecg", 0.0),
            "temperature": self.fill_value.get("temperature", 0.0),
        }

        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs
        )

        # load the file with pandas
        raw = read_csv(file, **self.read_csv_kwargs)

        # update the to_datetime_kwargs based on tz_name.  tz_name==None (utc=False)
        self.to_datetime_kw.update({"utc": tz_name is not None})

        # convert time column to a datetime column. Give a unique name so we shouldnt overwrite
        raw[self.time_col_name] = to_datetime(
            raw[self.time_col_name], **self.to_datetime_kw
        )

        # convert timestamps if necessary
        if tz_name is not None:
            # convert, and then remove the timezone so its naive again, but now in local time
            raw[self.time_col_name] = raw[self.time_col_name].dt.tz_convert(tz_name)
        
        if self.trim is not None:
            raw = self.trim_data(raw, kwargs.get("trim_start"), kwargs.get("trim_end"), tz_name)

        # now handle data gaps and second level timestamps, etc
        # raw, fs = self.handle_timestamp_inconsistency(raw, fill_values)

        # get the time values and convert to seconds
        time = (
            raw[self.time_col_name].astype(int).values / 1e9
        )  # int gives ns, convert to s

        data = {}
        # grab the data we expect
        for dstream in self.column_names:
            try:
                data[dstream] = raw[self.column_names[dstream]].values
            except KeyError:
                warn(
                    f"Data stream {dstream} specified in column names but all columns {self.column_names[dstream]} not found in the read data. Skipping."
                )
                continue

        # now handle data gaps and second level timestamps, etc
        fs, time, results = self.handle_timestamp_inconsistency_np(
            fill_values, time, data
        )

        # setup the results dictionary
        results.update(
            {
                self._time: time,
                "fs": fs,
            }
        )

        # convert accel data
        for k, conv_factor in self.raw_conversions.items():
            try:
                results[k] /= conv_factor
            except KeyError:
                continue

        return results
