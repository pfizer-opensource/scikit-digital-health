"""
Unit tests for wear detection algorithms

Lukas Adamowicz
Pfizer DMTI 2021
"""
import pytest
import numpy as np

from skimu.preprocessing.wear_detection import _modify_wear_times


class TestWearDetection:
    @pytest.mark.parametrize(
        ("case", "setup", "ship"),
        (
                (1, False, [0, 0]),
                (1, True, [12, 12]),
                (2, False, [0, 0]),
                (2, True, [12, 12]),
                (3, False, [0, 0]),
                (3, True, [12, 12]),
                (4, False, [0, 0]),
                (4, True, [12, 12])
        )
    )
    def test_modifiction1(self, case, setup, ship, sample_nonwear_data):
        wskip = 15  # minutes

        nonwear, t_starts, t_stops = sample_nonwear_data(case, wskip, setup, ship)

        starts, stops = _modify_wear_times(nonwear, wskip, setup, ship)

        assert np.allclose(starts, t_starts)
        assert np.allclose(stops, t_stops)
