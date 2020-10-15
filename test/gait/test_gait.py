"""
Testing of gait module functions and classes

Lukas Adamowicz
2020, Pfizer DMTI
"""
import pytest
from numpy import allclose, arange

from ..base_conftest import *

from skimu.gait import Gait
from skimu.gait.gait import LowFrequencyError
from skimu.gait.get_gait_classification import get_gait_classification_lgbm
from skimu.gait.get_gait_bouts import get_gait_bouts
from skimu.gait.get_gait_events import get_gait_events
from skimu.gait.get_strides import get_strides
from skimu.gait.get_bout_metrics_delta_h import get_bout_metrics_delta_h


class TestGetGaitClassificationLGBM:
    def test(self, sample_accel, sample_fs, sample_gait_classification_truth):
        b_gait = get_gait_classification_lgbm(sample_accel, sample_fs)

        assert b_gait.sum() == 13020
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
            'IC',
            'FC',
            'delta h',
            'b valid cycle',
            'PARAM:stride time',
            'PARAM:stance time',
            'PARAM:swing time',
            'PARAM:step time',
            'PARAM:initial double support',
            'PARAM:terminal double support',
            'PARAM:double support',
            'PARAM:single support',
            'PARAM:step length',
            'PARAM:stride length',
            'PARAM:gait speed',
            'PARAM:cadence',
            'PARAM:intra-step covariance - V',
            'PARAM:intra-stride covariance - V',
            'PARAM:harmonic ratio - V',
            'PARAM:stride SPARC',
            'BOUTPARAM:phase coordination index',
            'BOUTPARAM:gait symmetry index',
            'BOUTPARAM:step regularity - V',
            'BOUTPARAM:stride regularity - V',
            'BOUTPARAM:autocovariance symmetry - V',
            'BOUTPARAM:regularity index - V'
        ]
        cls.sample_data_keys.extend([
            'height'
        ])

        cls.process = Gait(
            use_cwt_scale_relation=True,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            max_stride_time=2.25,
            loading_factor=0.2,
            height_factor=0.53,
            leg_length=False,
            filter_order=4,
            filter_cutoff=20.0
        )

    def test_leg_length_warning(self, get_sample_data):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )
        data['height'] = None

        with pytest.warns(UserWarning):
            self.process._predict(**data)

    def test_sample_rate_error(self, get_sample_data):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )
        data['time'] = arange(0, 300, 0.5)

        with pytest.raises(LowFrequencyError):
            self.process._predict(**data)

    def test_gait_predictions_error(self, get_sample_data):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )
        data['gait_pred'] = arange(0, 1, 0.1)

        with pytest.raises(ValueError):
            self.process._predict(**data)
