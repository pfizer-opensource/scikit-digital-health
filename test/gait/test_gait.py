"""
Testing of gait module functions and classes

Lukas Adamowicz
2020, Pfizer DMTI
"""
import pytest
from numpy import allclose, arange, random
from scipy.interpolate import interp1d

from ..base_conftest import *

from skimu.gait import Gait
from skimu.gait.gait import LowFrequencyError
from skimu.gait.get_gait_classification import get_gait_classification_lgbm
from skimu.gait.get_gait_bouts import get_gait_bouts
from skimu.gait.get_gait_events import get_gait_events
from skimu.gait.get_strides import get_strides
from skimu.gait import gait_metrics
from skimu.gait.gait_metrics.gait_metrics import _autocovariancefunction, _autocovariance


class TestGetGaitClassificationLGBM:
    def test(self, sample_accel, sample_fs, get_gait_classification_truth):
        b_gait = get_gait_classification_lgbm(None, sample_accel, sample_fs)
        b_gait_truth = get_gait_classification_truth(sample_fs)

        assert b_gait.sum() == 13020
        assert allclose(b_gait, b_gait_truth)

    def test_20hz(self, sample_accel, sample_fs, get_gait_classification_truth):
        # downsample to 20hz
        f = interp1d(
            arange(0, sample_accel.shape[0] / sample_fs, 1 / sample_fs),
            sample_accel,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate',
            axis=0
        )
        acc_ds = f(arange(0, sample_accel.shape[0] / sample_fs, 1 / 20.0))

        b_gait = get_gait_classification_lgbm(None, acc_ds, 20.0)
        b_gait_truth = get_gait_classification_truth(20.0)

        assert b_gait.sum() == 1860
        assert allclose(b_gait, b_gait_truth)

    def test_pred_size_error(self, sample_accel):
        with pytest.raises(ValueError):
            get_gait_classification_lgbm(random.rand(50) > 0.5, sample_accel, 50.0)

    @pytest.mark.parametrize('pred', (True, False, 1, -135098135, 1.513e-600))
    def test_pred_single_input(self, pred, sample_accel):
        b_gait = get_gait_classification_lgbm(pred, sample_accel, 32.125)

        assert all(b_gait)

    def test_pred_array_input(self, sample_accel):
        pred = random.rand(sample_accel.shape[0]) < 0.5
        b_gait = get_gait_classification_lgbm(pred, sample_accel, 55.0)

        assert b_gait is pred
        assert allclose(b_gait, pred)


class TestGetGaitBouts:
    @pytest.mark.parametrize('case', (1, 2, 3, 4))
    def test(self, get_bgait_samples_truth, case):
        bgait, dt, max_sep, min_time, bouts = get_bgait_samples_truth(case)

        pred_bouts = get_gait_bouts(bgait, dt, max_sep, min_time)

        assert allclose(pred_bouts, bouts)


class TestGetGaitEvents:
    @pytest.mark.parametrize('sign', (1, -1))
    def test(self, sign, sample_fs, get_sample_bout_accel, get_contact_truth):
        accel, axis, acc_sign = get_sample_bout_accel(sample_fs)
        ic_truth, fc_truth = get_contact_truth(bout=1)  # index starts at 1 for this

        o_scale = round(0.4 / (2 * 1.25 / sample_fs)) - 1

        ic, fc, _ = get_gait_events(
            sign * accel[:, axis],
            1 / sample_fs,
            sign * acc_sign,
            o_scale, 4, 20.0, True
        )

        assert allclose(ic, ic_truth)
        assert allclose(fc, fc_truth)


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
            prov_leg_length=False,
            filter_order=4,
            filter_cutoff=20.0
        )

    def test_leg_length_factor(self):
        g = Gait(prov_leg_length=True, height_factor=0.53)

        assert g.height_factor == 1.0

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

    def test_add_metrics(self):
        g = Gait()
        g._params = []  # reset for easy testing

        g.add_metrics([gait_metrics.StrideTime, gait_metrics.StepTime])
        g.add_metrics(gait_metrics.PhaseCoordinationIndex)

        assert g._params == [
            gait_metrics.StrideTime,
            gait_metrics.StepTime,
            gait_metrics.PhaseCoordinationIndex
        ]

    def test_add_metrics_error(self):
        g = Gait()

        with pytest.raises(ValueError):
            g.add_metrics([list, Gait])

        with pytest.raises(ValueError):
            g.add_metrics(Gait)
