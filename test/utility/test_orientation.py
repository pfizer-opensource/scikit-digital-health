import pytest
from numpy import abs, mean, allclose

from skdh.utility.orientation import correct_accelerometer_orientation


class TestCorrectAccelerometerOrientation:
    def test(self, dummy_rotated_accel):
        x_orig, x_rot = dummy_rotated_accel

        x_corr = correct_accelerometer_orientation(x_rot, 2, None)

        # verify that this has reduced the mis-orientation significantly
        assert mean(abs(x_corr - x_orig)) < 0.1 * mean(abs(x_rot - x_orig))

        # test without providing vertical axis
        x_corr2 = correct_accelerometer_orientation(x_rot, None, None)
        assert allclose(x_corr2, x_corr)

        # test with providing ap axis
        x_corr3 = correct_accelerometer_orientation(x_rot, 2, 0)
        assert allclose(x_corr3, x_corr)

    def test_vaxis_range_error(self):
        with pytest.raises(ValueError):
            correct_accelerometer_orientation(None, 4, None)
        with pytest.raises(ValueError):
            correct_accelerometer_orientation(None, -1, None)

    def test_apaxis_range_error(self):
        with pytest.raises(ValueError):
            correct_accelerometer_orientation(None, 2, 4)
        with pytest.raises(ValueError):
            correct_accelerometer_orientation(None, 2, -1)
