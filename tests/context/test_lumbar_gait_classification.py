import pytest
from numpy import allclose, array

from skdh.context import PredictGaitLumbarLgbm
from skdh.utility.exceptions import LowFrequencyError


class TestPredictGaitLumbarLgbm:
    def test_50hz(self, gait_input_50):
        t, acc = gait_input_50

        proc = PredictGaitLumbarLgbm()
        proc._in_pipeline = True  # set so that we get all results

        kw, res = proc.predict(time=t, accel=acc, fs=50.0)

        # make sure they are starting on 3s window multiples
        assert allclose(res["Gait Bout Start Index"] % 150, 0)
        assert allclose(res["Gait Bout Stop Index"] % 150, 0)

        assert allclose(res["Gait Bout Start Index"], [600, 900, 2550])
        assert allclose(res["Gait Bout Stop Index"], [750, 2400, 3450])

        # check kw shape
        assert kw["gait_bouts"].shape == (3, 2)

    def test_warnings_errors(self, gait_input_50):
        t, acc = gait_input_50
        proc = PredictGaitLumbarLgbm()

        with pytest.raises(LowFrequencyError):
            proc.predict(time=None, accel=None, fs=15.0)

        with pytest.warns():
            proc.predict(time=t, accel=acc, fs=25.0)
