"""
Testing of gait module functions and classes

Lukas Adamowicz
2020, Pfizer DMTI
"""
import pytest
from numpy import allclose

from PfyMU.gait import Gait
from PfyMU.gait.get_gait_classification import get_gait_classification_lgbm
from PfyMU.gait.get_gait_bouts import get_gait_bouts
from PfyMU.gait.get_gait_events import get_gait_events
from PfyMU.gait.get_strides import get_strides
from PfyMU.gait.get_gait_metrics import get_gait_metrics_initial, get_gait_metrics_final


class TestGetGaitClassificationLGBM:
    def test(self, sample_accel, sample_fs):
        b_gait = get_gait_classification_lgbm(sample_accel, sample_fs)

        assert True


class TestGetGaitBouts:
    @pytest.mark.parametrize('case', (1, 2, 3, 4))
    def test(self, get_bgait_samples_truth, case):
        bgait, dt, max_sep, min_time, bouts = get_bgait_samples_truth(case)

        pred_bouts = get_gait_bouts(bgait, dt, max_sep, min_time)

        assert allclose(pred_bouts, bouts)


