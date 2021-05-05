"""
Test activity functionality

Lukas Adamowicz
Pfizer DMTI
2021
"""
import pytest
from numpy import allclose, isclose

from skimu.activity.core import get_activity_bouts

from skimu import Pipeline
from skimu.read import ReadBin
from skimu.preprocessing import CalibrateAccelerometer
from skimu.preprocessing import DetectWear
from skimu.activity import ActivityLevelClassification
from skimu.sleep import Sleep


class TestGetActivityBouts:
    @pytest.mark.parametrize(
        "boutdur,boutmetric", [[i, j] for i in range(1, 3) for j in range(1, 6)]
    )
    def test_no_mvpa_at_ends(self, boutdur, boutmetric, get_sample_activity_bout_data1):
        sample_acc_metric, mvpa_time_true = get_sample_activity_bout_data1(boutmetric, boutdur)

        mvpa_time = get_activity_bouts(
            sample_acc_metric,
            lower_thresh=0.1,
            upper_thresh=1e5,
            wlen=5,
            boutdur=boutdur,
            boutcrit=0.8,
            closedbout=False,
            boutmetric=boutmetric
        )

        assert isclose(mvpa_time, mvpa_time_true)

    @pytest.mark.parametrize(
        "boutdur,boutmetric", [[i, j] for i in range(1, 3) for j in range(1, 6)]
    )
    def test_mvpa_at_ends(self, boutdur, boutmetric, get_sample_activity_bout_data2):
        sample_acc_metric, mvpa_time_true = get_sample_activity_bout_data2(boutmetric, boutdur)

        mvpa_time = get_activity_bouts(
            sample_acc_metric,
            lower_thresh=0.1,
            upper_thresh=1e5,
            wlen=5,
            boutdur=boutdur,
            boutcrit=0.8,
            closedbout=False,
            boutmetric=boutmetric
        )

        assert isclose(mvpa_time, mvpa_time_true)


# TODO update to an actual test
class TestMVPActivityClassification:
    def test(self):
        # pipe = Pipeline()
        # pipe.add(ReadBin(bases=[0, 12], periods=[24, 24]))
        # pipe.add(CalibrateAccelerometer())
        # pipe.add(Sleep())
        # pipe.add(DetectWear())
        #
        # act = ActivityLevelClassification(
        #         bout_metric=4,
        #         cutpoints="migueles_wrist_adult"
        #     )
        # act.setup_plotting("/Users/lukasadamowicz/Downloads/STRYDE/skimu_results/activity_plot.html")
        #
        # pipe.add(
        #     act,
        #     save_results=True,
        #     save_name="/Users/lukasadamowicz/Downloads/STRYDE/skimu_results/activity_results.csv"
        # )
        #
        # pipe.run(file="/Users/lukasadamowicz/Downloads/STRYDE/stryde/100111980001_GNACTV_LeftWrist.bin")
        assert True
