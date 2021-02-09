"""
Testing Calibration process
"""
from numpy import allclose, array, around, mean, abs
from numpy.linalg import norm

from skimu.read import ReadBin
from skimu.preprocessing import CalibrateAccelerometer


class TestCalibration:
    def test_no_temp(self, sample_data_long):
        t, acc, true_scale, true_offset = sample_data_long

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer()
        cal_res = cal.predict(t, acc, apply=True)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=5e-5)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)

    def test_w_temp(self, sample_data_temp):
        t, acc, temp, true_scale, true_offset, true_temp_scale = sample_data_temp

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer()
        cal_res = cal.predict(t, acc, apply=True, temperature=temp)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=5e-5)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)
        # pretty lax again, since the values are very small themselves
        assert allclose(cal_res["temperature scale"], true_temp_scale, atol=2e-4)
