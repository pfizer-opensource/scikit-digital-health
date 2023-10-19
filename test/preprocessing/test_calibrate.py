from psutil import virtual_memory
import pytest
from numpy import around, mean, abs, arange, array, zeros, allclose
from numpy.linalg import norm

from skdh.preprocessing import CalibrateAccelerometer


class TestCalibrateAccelerometer:
    @pytest.mark.slow
    @pytest.mark.skipif(virtual_memory().available < 90e6, reason="Insufficient memory")
    def test_no_temp(self, dummy_long_data):
        t, acc, true_scale, true_offset = dummy_long_data

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer(min_hours=12)
        cal_res = cal.predict(time=t, accel=acc, apply=True)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        # assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=1e-4)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)

    @pytest.mark.slow
    @pytest.mark.skipif(virtual_memory().available < 90e6, reason="Insufficient Memory")
    def test_w_temp(self, dummy_temp_data):
        t, acc, temp, true_scale, true_offset, true_temp_scale = dummy_temp_data

        error_start = around(mean(abs(norm(acc, axis=1) - 1)), decimals=5)

        cal = CalibrateAccelerometer(min_hours=12)
        cal_res = cal.predict(time=t, accel=acc, apply=True, temperature=temp)

        error_end = around(mean(abs(norm(cal_res["accel"], axis=1) - 1)), decimals=5)

        # assert error_end < error_start
        assert allclose(cal_res["scale"], true_scale, rtol=1e-4)
        # pretty lax here, since offset can be different and still give good values
        assert allclose(cal_res["offset"], true_offset, atol=2e-4)
        # pretty lax again, since the values are very small themselves
        assert allclose(cal_res["temperature scale"], true_temp_scale, atol=2e-4)

    def test_under_12h_data(self, np_rng):
        t = arange(0, 3600 * 1, 1 / 50)
        a = np_rng.random((t.size, 3))

        # minimum hours should get auto bumped to 12
        cal = CalibrateAccelerometer(min_hours=4)
        assert cal.min_hours == 12, "Hours not rounded up to multiple of 12"

        with pytest.warns(UserWarning) as record:
            cal.predict(time=t, accel=a)

        assert len(record) == 1  # only 1 warning before returning
        assert "Less than 12 hours of data" in record[0].message.args[0]

    def test_all_motion(self, np_rng):
        t = arange(0, 3600 * 15, 1 / 50)
        a = (np_rng.random((t.size, 3)) - 0.5) * 0.25 + array([1, 0, 0])

        cal = CalibrateAccelerometer(min_hours=12, sd_criteria=0.01)

        with pytest.warns(UserWarning) as record:
            cal.predict(time=t, accel=a)

        assert len(record) == 1
        assert "insufficient non-movement data available" in record[0].message.args[0]

    def test_bad_sphere(self):
        """
        Testing not enough points around the full sphere
        """
        t = arange(0, 3600 * 15, 1 / 50)
        a = zeros((t.size, 3)) + array([0.95, 0.001, -0.05])

        cal = CalibrateAccelerometer(min_hours=12)

        with pytest.warns(UserWarning) as record:
            cal.predict(time=t, accel=a)

        assert len(record) == 1
        assert "insufficient non-movement data available" in record[0].message.args[0]
