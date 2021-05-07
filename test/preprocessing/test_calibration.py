"""
Testing Calibration process
"""
import pytest
import psutil
from numpy import allclose, around, mean, abs, arange, random, array, zeros
from numpy.linalg import norm

from skimu.preprocessing import CalibrateAccelerometer


class TestCalibration:
    @pytest.mark.skipif(
        psutil.virtual_memory().available < 90e6, reason="Not enough memory"
    )
    def test_no_temp(self, sample_data_long):
        t, acc, true_scale, true_offset = sample_data_long

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer(min_hours=12)
        cal_res = cal.predict(t, acc, apply=True)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=5e-5)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)

    @pytest.mark.skipif(
        psutil.virtual_memory().available < 90e6, reason="Not enough memory"
    )
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

    def test_under_12h_data(self):
        t = arange(0, 3600 * 11, 1 / 50)
        a = random.random((t.size, 3))

        # minimum hours should get set to 12
        cal = CalibrateAccelerometer(min_hours=10)

        assert cal.min_hours == 12

        with pytest.warns(UserWarning):
            cal.predict(t, a)

    def test_all_motion(self):
        t = arange(0, 3600 * 15, 1 / 50)
        a = (random.random((t.size, 3)) - 0.5) * 0.25 + array([1, 0, 0])

        cal = CalibrateAccelerometer(min_hours=12, sd_criteria=0.01)

        with pytest.warns(UserWarning):
            cal.predict(t, a)

    def test_bad_sphere(self):
        """
        Testing not enough points around the full sphere
        """
        t = arange(0, 3600 * 15, 1 / 50)
        a = zeros((t.size, 3)) + array([0.95, 0.001, -0.05])

        cal = CalibrateAccelerometer(min_hours=12)

        with pytest.warns(UserWarning):
            cal.predict(t, a)
