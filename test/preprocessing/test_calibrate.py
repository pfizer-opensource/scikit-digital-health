from psutil import virtual_memory
import pytest
from numpy import around, mean, abs, allclose
from numpy.linalg import norm

from skimu.preprocessing import CalibrateAccelerometer


class TestCalibrateAccelerometer:
    @pytest.mark.skipif(virtual_memory().available < 90e6, reason="Insufficient memory")
    def test_no_temp(self, dummy_long_data):
        t, acc, true_scale, true_offset = dummy_long_data

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer(min_hours=12)
        cal_res = cal.predict(t, acc, apply=True)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=5e-5)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)

    @pytest.mark.skipif(virtual_memory().available < 90e6, reason="Insufficient Memory")
    def test_w_temp(self, dummy_temp_data):
        t, acc, temp, true_scale, true_offset, true_temp_scale = dummy_temp_data

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer(min_hours=12)
        cal_res = cal.predict(t, acc, apply=True, temperature=temp)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=5e-5)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)
        # pretty lax again, since the values are very small themselves
        assert allclose(cal_res["temperature scale"], true_temp_scale, atol=2e-4)
