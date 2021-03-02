"""
Test activity functionality

Lukas Adamowicz
Pfizer DMTI
2021
"""
import pytest
from numpy import allclose, isclose

from skimu.activity.core import get_activity_bouts


class TestGetActivityBouts:
    @pytest.mark.parametrize("boutmetric", (1, 2, 3, 4, 5))
    def test(self, boutmetric, get_sample_activity_bout_data):
        sample_acc_metric, mvpa_time_true = get_sample_activity_bout_data(boutmetric)

        mvpa_time = get_activity_bouts(
            sample_acc_metric,
            mvpa_thresh=0.1,
            wlen=5,
            boutdur=1,
            boutcrit=0.8,
            closedbout=False,
            boutmetric=boutmetric
        )

        assert isclose(mvpa_time, mvpa_time_true)
