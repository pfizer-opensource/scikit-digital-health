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
    @pytest.mark.parametrize(
        ("boutdur", "boutmetric"),
        (
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5)
        )
    )
    def test(self, boutdur, boutmetric, get_sample_activity_bout_data):
        sample_acc_metric, mvpa_time_true = get_sample_activity_bout_data(boutmetric, boutdur)

        mvpa_time = get_activity_bouts(
            sample_acc_metric,
            mvpa_thresh=0.1,
            wlen=5,
            boutdur=boutdur,
            boutcrit=0.8,
            closedbout=False,
            boutmetric=boutmetric
        )

        assert isclose(mvpa_time, mvpa_time_true)
