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
    @pytest.mark.parametrize("boutdur", (1, 2))
    def test_no_mvpa_at_ends(self, boutdur, get_sample_activity_bout_data1):
        for boutmetric in range(1, 6):
            sample_acc_metric, mvpa_time_true = get_sample_activity_bout_data1(boutmetric, boutdur)

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

    @pytest.mark.parametrize("boutdur", (1, 2))
    def test_mvpa_at_ends(self, boutdur, get_sample_activity_bout_data2):
        for boutmetric in range(1, 6):
            sample_acc_metric, mvpa_time_true = get_sample_activity_bout_data2(boutmetric, boutdur)

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