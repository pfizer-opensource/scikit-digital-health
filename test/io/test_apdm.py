import pytest
import h5py
from tempfile import NamedTemporaryFile

from numpy import allclose

from skdh.io import ReadApdmH5
from skdh.io.apdm import SensorNotFoundError


class TestReadBin:
    def test(self, apdm_file):
        res = ReadApdmH5("Lumbar", gravity_acceleration=9.81).predict(apdm_file)

        lumbar_sens = "XI-010284"
        with h5py.File(apdm_file) as f:
            acc = f["Sensors"][lumbar_sens]["Accelerometer"][()] / 9.81
            time = f["Sensors"][lumbar_sens]["Time"][()] / 1e6  # to seconds
            gyro = f["Sensors"][lumbar_sens]["Gyroscope"][()]
            temp = f["Sensors"][lumbar_sens]["Temperature"][()]

        assert allclose(res["accel"], acc)
        assert allclose(res["time"] - time[0], time - time[0])
        assert allclose(res["gyro"], gyro)
        assert allclose(res["temperature"], temp)

    def test_extension(self):
        with NamedTemporaryFile(suffix=".abc") as tmpf:
            with pytest.warns(UserWarning, match=r"expected \[.h5\]"):
                with pytest.raises(Exception):
                    ReadApdmH5("Lumbar").predict(tmpf.name)

    def test_bad_sensor(self, apdm_file):
        with pytest.raises(SensorNotFoundError):
            ReadApdmH5("badSensor", gravity_acceleration=9.81).predict(apdm_file)
