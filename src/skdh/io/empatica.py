from warnings import warn

from avro.datafile import DataFileReader
from avro.io import DatumReader
from numpy import (
    round,
    arange,
    vstack,
    ascontiguousarray,
    isclose,
    full,
    argmin,
    abs,
    nan,
    float64,
)

from skdh import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file
from skdh.utility.internal import apply_resample


class ReadEmpaticaAvro(BaseProcess):
    """
    Read Empatica data from an avro file.

    Parameters
    ----------
    resample_to_accel : bool, optional
        Resample any additional data streams to match the accelerometer data stream.
        Default is True.
    """

    def __init__(self, resample_to_accel=True):
        super().__init__(resample_to_accel=resample_to_accel)

        self.resample_to_accel = resample_to_accel

    def get_accel(self, raw_accel_dict, results_dict, key):
        """
        Get the raw acceleration data from the avro file record.

        Parameters
        ----------
        raw_accel_dict : dict
            The record from the avro file for a raw data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        # sampling frequency
        fs = raw_accel_dict["samplingFrequency"]

        # timestamp start
        ts_start = raw_accel_dict["timestampStart"] / 1e6  # convert to seconds

        # imu parameters for scaling to actual values
        phys_min = raw_accel_dict["imuParams"]["physicalMin"]
        phys_max = raw_accel_dict["imuParams"]["physicalMax"]
        dig_min = raw_accel_dict["imuParams"]["digitalMin"]
        dig_max = raw_accel_dict["imuParams"]["digitalMax"]

        # raw acceleration data
        accel = ascontiguousarray(
            vstack((raw_accel_dict["x"], raw_accel_dict["y"], raw_accel_dict["z"])).T
        )

        # scale the raw acceleration data to actual values
        accel = (accel - dig_min) / (dig_max - dig_min) * (
            phys_max - phys_min
        ) + phys_min

        # create the timestamp array using ts_start, fs, and the number of samples
        time = arange(ts_start, ts_start + accel.shape[0] / fs, 1 / fs)[
            : accel.shape[0]
        ]

        if time.size != accel.shape[0]:
            raise ValueError("Time does not have enough samples for accel array")

        # use special names here so we can just update dictionary later for returning
        results_dict[key] = {self._time: time, "fs": fs, self._acc: accel}

    def get_gyroscope(self, raw_gyro_dict, results_dict, key):
        """
        Get the raw gyroscope data from the avro file record.

        Parameters
        ----------
        raw_gyro_dict : dict
            The record from the avro file for a raw gyroscope data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_gyro_dict["x"]:
            return

        # sampling frequency
        fs = round(raw_gyro_dict["samplingFrequency"], decimals=3)
        # timestamp start
        ts_start = raw_gyro_dict["timestampStart"] / 1e6  # convert to seconds
        # imu parameters for scaling to actual values
        phys_min = raw_gyro_dict["imuParams"]["physicalMin"]
        phys_max = raw_gyro_dict["imuParams"]["physicalMax"]
        dig_min = raw_gyro_dict["imuParams"]["digitalMin"]
        dig_max = raw_gyro_dict["imuParams"]["digitalMax"]

        # raw gyroscope data
        gyro = ascontiguousarray(
            vstack((raw_gyro_dict["x"], raw_gyro_dict["y"], raw_gyro_dict["z"])).T
        )

        # scale the raw gyroscope data to actual values
        gyro = (gyro - dig_min) / (dig_max - dig_min) * (phys_max - phys_min) + phys_min

        # create the timestamp array using ts_start, fs, and the number of samples
        time = arange(ts_start, ts_start + gyro.shape[0] / fs, 1 / fs)[: gyro.shape[0]]

        if time.size != gyro.shape[0]:
            raise ValueError("Time does not have enough samples for gyro array")

        results_dict[key] = {self._time: time, "fs": fs, "values": gyro}

    def get_values_1d(self, raw_dict, results_dict, key):
        """
        Get the raw 1-dimensional values data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for a raw 1-dimensional values data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["values"]:
            return

        # sampling frequency
        fs = round(raw_dict["samplingFrequency"], decimals=3)
        # timestamp start
        ts_start = raw_dict["timestampStart"] / 1e6  # convert to seconds

        # raw values data
        values = ascontiguousarray(raw_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + values.size / fs, 1 / fs)[: values.shape[0]]

        if time.size != values.shape[0]:
            raise ValueError(f"Time does not have enough samples for {key} array")

        results_dict[key] = {self._time: time, "fs": fs, "values": values}

    @staticmethod
    def get_systolic_peaks(raw_dict, results_dict, key):
        """
        Get the systolic peaks data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for systolic peaks data.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["peaksTimeNanos"]:
            return

        peaks = (
            ascontiguousarray(raw_dict["peaksTimeNanos"]) / 1e9
        )  # convert to seconds

        results_dict[key] = {"values": peaks}

    def get_steps(self, raw_dict, results_dict, key):
        """
        Get the raw steps data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for raw steps data.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["values"]:
            return

        # sampling frequency
        fs = round(raw_dict["samplingFrequency"], decimals=3)

        # timestamp start
        ts_start = raw_dict["timestampStart"] / 1e6  # convert to seconds

        # raw steps data
        steps = ascontiguousarray(raw_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + steps.size / fs, 1 / fs)[: steps.size]

        if time.size != steps.size:
            raise ValueError("Time does not have enough samples for steps array")

        results_dict[key] = {self._time: time, "fs": fs, "values": steps}

    def handle_resampling(self, streams):
        """
        Handle resampling of data streams. Data will be resampled to match the
        accelerometer data stream.

        Parameters
        ----------
        streams : dict
            Dictionary containing the data streams

        Returns
        -------
        rs_streams : dict
            Dictionary containing the resampled data streams
        """
        # remove accelerometer data stream
        acc_dict = streams.pop(self._acc)
        # remove keys that we can't resample
        rs_streams = {
            d: streams.pop(d) for d in ["systolic_peaks", "steps"] if d in streams
        }

        # iterate over remaining streams and resample them
        for name, stream in streams.items():
            if stream["values"] is None:
                continue

            # check that the stream doesn't start significantly later than accelerometer
            if (dt := (stream["time"][0] - acc_dict["time"][0])) > 1:
                warn(
                    f"Data stream {name} starts more than 1 second ({dt}s) after "
                    f"the accelerometer. Data will be filled with the first (and "
                    f"last) value as needed."
                )

            # check if we need to resample
            if isclose(stream["fs"], acc_dict["fs"], atol=1e-3):
                # create the new shape to match the accel shape
                new_shape = list(stream["values"].shape)
                new_shape[0] = acc_dict[self._acc].shape[0]
                # create the full length data shape
                rs_streams[name] = full(new_shape, nan, dtype=float64)
                # get the indices for the stream data
                i1 = argmin(abs(acc_dict["time"] - stream["time"][0]))
                i2 = i1 + stream["time"].size
                # put the stream values into the correct size array
                rs_streams[name][i1:i2] = stream["values"][
                    : stream["values"].shape[0] - (i2 - acc_dict["time"].size)
                ]
                # put the first value in the beginning as needed
                rs_streams[name][:i1] = stream["values"][0]
                # put the last value in the end as needed
                rs_streams[name][i2:] = stream["values"][-1]
                continue

            # resample the stream
            _, (stream_rs,) = apply_resample(
                time=stream["time"],
                time_rs=acc_dict["time"],
                data=(stream["values"],),
                aa_filter=True,
                fs=stream["fs"],
            )
            rs_streams[name] = stream_rs

        # add accelerometer data back in
        rs_streams.update(acc_dict)

        return rs_streams

    def get_datastreams(self, raw_record):
        """
        Get the various data streams from the raw avro file record.

        Parameters
        ----------
        raw_record : dict
            The raw avro file record

        Returns
        -------
        data_streams : dict
            Dictionary containing the data streams
        """
        fn_map = {
            "accelerometer": (self._acc, self.get_accel),
            "gyroscope": (self._gyro, self.get_gyroscope),
            "eda": ("eda", self.get_values_1d),
            "temperature": (self._temp, self.get_values_1d),
            "bvp": ("bvp", self.get_values_1d),
            "systolicPeaks": ("systolic_peaks", self.get_systolic_peaks),
            "steps": ("steps", self.get_steps),
        }

        raw_data_streams = {}
        for full_name, (stream_name, fn) in fn_map.items():
            fn(raw_record[full_name], raw_data_streams, stream_name)

        # handle re-scaling if desired, will handle re-formatting as well
        if self.resample_to_accel:
            data_streams = self.handle_resampling(raw_data_streams)
        else:
            # remove the accel dictionary to form the basis for the return dictionary
            data_streams = raw_data_streams.pop(self._acc)
            data_streams.update(
                raw_data_streams
            )  # add rest of data streams, keeping as dicts

        return data_streams

    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(extension=".avro", check_size=True)
    def predict(self, *, file, tz_name=None, **kwargs):
        """
        Read the input .avro file.

        Parameters
        ----------
        file : {path-like, str}
            The path to the input file.
        tz_name : {None, optional}
            IANA time-zone name for the recording location. If not provided, timestamps
            will represent local time naively. This means they will not account for
            any time changes due to Daylight Saving Time.

        Returns
        -------
        results : dict
            Dictionary containing the data streams from the file. See Notes
            for different output options.

        Notes
        -----
        There are two output formats, based on if `resample_to_accel` is True or False.
        If True, all available data streams except for `systolic_peaks` and `steps`
        are resampled to match the accelerometer data stream, which results in their
        values being present in the top level of the `results` dictionary, ie
        `results['gyro']`, etc.

        If False, everything except accelerometer will be present in dictionaries
        containing the keys `time`, `fs`, and `values`, and the top level will be these
        dictionaries plus the accelerometer data (keys `time`, `fs`, and `accel`).

        `systolic_peaks` will always be a dictionary of the form `{'systolic_peaks': array}`.
        """
        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs
        )

        reader = DataFileReader(open(file, "rb"), DatumReader())
        records = []
        for record in reader:
            records.append(record)
        reader.close()

        # get the timezone offset
        tz_offset = records[0]["timezone"]  # in seconds

        # as needed, deviceSn, deviceModel

        # get the data streams
        results = self.get_datastreams(records[0]["rawData"])

        # update the timestamps to be local. Do this as we don't have an actual
        # timezone from the data.
        if tz_name is None:
            results["time"] += tz_offset

            for k in results:
                if k == "time":
                    continue
                if (
                    isinstance(results[k], dict)
                    and "time" in results[k]
                    and results[k]["time"] is not None
                ):
                    results[k]["time"] += tz_offset
        # do nothing if we have the time-zone name, the timestamps are already
        # UTC

        # adjust systolic_peaks
        if "systolic_peaks" in results:
            results["systolic_peaks"]["values"] += tz_offset

        return results
