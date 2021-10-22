import pytest
from numpy import allclose, array

from skdh.gait.get_gait_classification import get_gait_classification_lgbm


class Test_get_gait_classification_lgbm:
    def test_50hz(self, gait_input_50):
        t, acc = gait_input_50

        starts, stops = get_gait_classification_lgbm(None, None, acc, 50.0)

        # make sure they are starting on 3s window multiples
        assert allclose(starts % 150, 0)
        assert allclose(stops % 150, 0)

        # actual values
        assert allclose(starts, [600, 900, 2550])
        assert allclose(stops, [750, 2400, 3450])

    def test_fs_error(self):
        with pytest.raises(ValueError):
            get_gait_classification_lgbm(None, None, None, 100.0)

    def test_provided(self):
        start_in = array([1, 2, 3])
        stop_in = array([5, 10, 15])

        starts, stops = get_gait_classification_lgbm(start_in, stop_in, None, 50.0)

        assert starts is start_in
        assert stops is stop_in
