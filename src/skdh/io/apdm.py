"""
GeneActiv reading process

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

import h5py
from pandas import Timestamp
from numpy import isclose

from skdh.base import BaseProcess, handle_process_returns
from skdh.io.base import check_input_file


class SensorNotFoundError(Exception):
    pass


class ReadApdmH5(BaseProcess):
    """
    Read a H5 file produced by the APDM software into memory. Acceleration values
    are returned in units of `g`.

    Parameters
    ----------
    sensor_location : str
        Sensor location to get data from. Looks at the `Label 0` key to find the
        desired sensor.
    localize_timestamps : bool, optional
        Convert timestamps to local time from UTC. Default is True. Uses APDM's
        timezone offset attribute for the sensor being extracted. Ignored if a
        time-zone name is provided (`tz_name`) in the `predict` method.
    gravity_acceleration : float, optional
        Acceleration due to gravity. Used to convert values to units of `g`.
        Default is 9.81 m/s^2.
    ext_error : {"warn", "raise", "skip"}, optional
        What to do if the file extension does not match the expected extension (.h5).
        Default is "warn". "raise" raises a ValueError. "skip" skips the file
        reading altogether and attempts to continue with the pipeline.

    Notes
    -----
    Given that APDM systems are typically designed to be used in-lab, and not
    collect significant amounts of data (ie less than a day), there is no
    day windowing available for APDM data.

    Some of the default options for locations are:
    - Left/Right Foot
    - Left/Right Lower Leg
    - Left/Right Upper Leg
    - Lumbar
    - Sternum
    """

    def __init__(
        self,
        sensor_location,
        localize_timestamps=True,
        gravity_acceleration=9.81,
        ext_error="warn",
    ):
        super().__init__(
            # kwargs
            sensor_location=sensor_location,
            localize_timestamps=localize_timestamps,
            gravity_acceleration=gravity_acceleration,
            ext_error=ext_error,
        )

        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")

        self.localize_time = localize_timestamps
        self.sens = sensor_location
        self.g = gravity_acceleration

    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(".h5", check_size=False)
    def predict(self, *, file, tz_name=None, **kwargs):
        """
        predict(*, file)

        Read the data from an APDM file, getting the data from the specified sensor.

        Parameters
        ----------
        file : {str, pathlib.Path}
            Path to the file to read. Must either be a string, or be able to be
            converted by `str(file)`.
        tz_name : {None, str}, optional
            IANA time-zone name for the recording location. If provided, and `localize_timestamps`
            is True, will check the offset matches that from the file. If not provided,
            will convert to local time as specified by `localize_timestamps`.

        Returns
        -------
        data : dict
            Dictionary of the data contained in the file.

        Raises
        ------
        ValueError
            If the file name is not provided.
        ValueError
            If the time-zone name offset does not match the offset in the file itself.
        FileNotFoundError
            If the file does not exist.
        skdh.io.SensorNotFoundError
            If the specified sensor name was not found.
        """
        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs
        )

        res = {}
        # read the file
        with h5py.File(file, "r") as f:
            sid = None  # sensor id
            for sens in f["Sensors"]:
                try:
                    sname = f["Sensors"][sens]["Configuration"].attrs["Label 0"]
                except (RuntimeError, KeyError):
                    # if the sensor has issues, still try to find in other sensors
                    continue
                if sname.decode("utf-8") == self.sens:
                    sid = sens
            if sid is None:
                raise SensorNotFoundError(f"Sensor {self.sens} was not found.")

            res[self._acc] = f["Sensors"][sid]["Accelerometer"][()] / self.g
            res[self._time] = f["Sensors"][sid]["Time"][()] / 1e6  # to seconds
            res[self._gyro] = f["Sensors"][sid]["Gyroscope"][()]
            res[self._temp] = f["Sensors"][sid]["Temperature"][()]

            # if we are converting to local time
            if tz_name is not None:
                # check that the file offset matches that from the tz_name
                file_offset = (
                    float(f["Sensors"][sid]["Configuration"].attrs["Timezone Offset"])
                    * 3600.0
                )

                tz_offset = (
                    Timestamp(res[self._time][0], unit="s", tz=tz_name)
                    .utcoffset()
                    .total_seconds()
                )

                if not isclose(tz_offset, file_offset):
                    raise ValueError(
                        "Timezone offset from the file does not match timezone "
                        "offset from the provided tz_name."
                    )

                # dont do any conversion since we have the time-zone name
            else:
                if self.localize_time:
                    offset_hours = float(
                        f["Sensors"][sid]["Configuration"].attrs["Timezone Offset"]
                    )
                    # convert to seconds
                    offset_sec = offset_hours * 3600.0

                    res[self._time] += offset_sec

        return res
