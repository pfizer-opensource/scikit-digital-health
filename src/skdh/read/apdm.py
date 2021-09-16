"""
GeneActiv reading process

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn
from pathlib import Path

import h5py

from skdh.base import BaseProcess


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
    gravity_acceleration : float, optional
        Acceleration due to gravity. Used to convert values to units of `g`.
        Default is 9.81 m/s^2.

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

    def __init__(self, sensor_location, gravity_acceleration=9.81):
        super().__init__(
            # kwargs
            sensor_location=sensor_location,
            gravity_acceleration=gravity_acceleration,
        )

        self.sens = sensor_location
        self.g = gravity_acceleration

    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from an APDM file, getting the data from the specified sensor.

        Parameters
        ----------
        file : {str, pathlib.Path}
            Path to the file to read. Must either be a string, or be able to be
            converted by `str(file)`.
        kwargs

        Returns
        -------
        data : dict
            Dictionary of the data contained in the file.

        Raises
        ------
        ValueError
            If the file name is not provided.
        FileNotFoundError
            If the file does not exist.
        skdh.read.SensorNotFoundError
            If the specified sensor name was not found.
        """
        if file is None:
            raise ValueError("`file` must not be None.")
        if not isinstance(file, str):
            file = str(file)
        if file[-2:] != "h5":
            warn("File extension is not expected '.h5'", UserWarning)
        if not Path(file).exists():
            raise FileNotFoundError(f"File {file} does not exist.")

        super().predict(expect_days=False, expect_wear=False, file=file, **kwargs)

        res = {}
        # read the file
        with h5py.File(file, "r") as f:
            sid = None  # sensor id
            for sens in f["Sensors"]:
                sname = f["Sensors"][sens]["Configuration"].attrs["Label 0"]
                if sname.decode("utf-8") == self.sens:
                    sid = sens
            if sens is None:
                raise SensorNotFoundError(f"Sensor {self.sens} was not found.")

            res[self._acc] = f["Sensors"][sid]["Accelerometer"][()] / self.g
            res[self._time] = f["Sensors"][sid]["Time"][()] / 1e6  # to seconds
            try:
                res[self._gyro] = f["Sensors"][sid]["Gyroscope"][()]
            except KeyError:
                # TODO change if processes start using gyroscope data.
                # leave as info for now since no algorithms use gyro data yet
                self.logger.info("No gyroscope data found.", UserWarning)
            try:
                res[self._temp] = f["Sensors"][sid]["Temperature"][()]
            except KeyError:
                warn("No temperature data found.", UserWarning)

        res["file"] = file
        kwargs.update(res)

        return (kwargs, None) if self._in_pipeline else kwargs
