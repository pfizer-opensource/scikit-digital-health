"""
Testing of gait module functions and classes

Lukas Adamowicz
2020, Pfizer DMTI
"""
import pytest
from numpy import allclose

from ..base_conftest import *

from PfyMU.gait import Gait
from PfyMU.gait.get_gait_classification import get_gait_classification_lgbm
from PfyMU.gait.get_gait_bouts import get_gait_bouts
from PfyMU.gait.get_gait_events import get_gait_events
from PfyMU.gait.get_strides import get_strides
from PfyMU.gait.get_bout_metrics_delta_h import get_bout_metrics_delta_h


class TestGetGaitClassificationLGBM:
    def test(self, sample_accel, sample_fs, sample_gait_classification_truth):
        b_gait = get_gait_classification_lgbm(sample_accel, sample_fs)

        assert b_gait.sum() == 15637
        assert allclose(b_gait, sample_gait_classification_truth)


class TestGetGaitBouts:
    @pytest.mark.parametrize('case', (1, 2, 3, 4))
    def test(self, get_bgait_samples_truth, case):
        bgait, dt, max_sep, min_time, bouts = get_bgait_samples_truth(case)

        pred_bouts = get_gait_bouts(bgait, dt, max_sep, min_time)

        assert allclose(pred_bouts, bouts)


class TestGetGaitEvents:
    def test(self):
        pass


class TestGait(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # override necessary attributes
        cls.sample_data_file = resolve_data_path('gait_data.h5', 'gait')
        cls.truth_data_file = resolve_data_path('gait_data.h5', 'gait')
        cls.truth_suffix = None
        cls.truth_data_keys = [
            'PARAM:stride time'
        ]
        cls.sample_data_keys.extend([
            'height'
        ])

        cls.process = Gait(
            use_cwt_scale_relation=True,
            min_bout_time=5.0,
            max_bout_separation_time=0.5,
            max_stride_time=2.25,
            loading_factor=0.2,
            height_factor=0.53,
            leg_length=False,
            filter_order=4,
            filter_cutoff=20.0
        )

    def test(self, get_sample_data, get_truth_data):
        super(TestGait, self).test(get_sample_data, get_truth_data)
