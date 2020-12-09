"""
Testing of gait module functions and classes

Lukas Adamowicz
2020, Pfizer DMTI
"""
import pytest
from numpy import allclose, arange, random, array, isnan, zeros

from ..base_conftest import *

from skimu.gait import Gait
from skimu.gait.gait import LowFrequencyError
from skimu.gait.get_gait_classification import get_gait_classification_lgbm, DimensionMismatchError
from skimu.gait.get_gait_bouts import get_gait_bouts
from skimu.gait.get_gait_events import get_gait_events
from skimu.gait.get_strides import get_strides
from skimu.gait import gait_metrics


class TestGetGaitClassificationLGBM:
    def test_50hz(self, get_sample_accel, get_gait_classification_truth):
        time, fs, accel = get_sample_accel(50.0)

        starts, stops = get_gait_classification_lgbm(None, accel, fs)
        starts_truth, stops_truth = get_gait_classification_truth(50)

        assert allclose(starts, starts_truth)
        assert allclose(stops, stops_truth)

    def test_20hz(self, get_sample_accel, get_gait_classification_truth):
        time, fs, accel = get_sample_accel(20.0)

        starts, stops = get_gait_classification_lgbm(None, accel, fs)
        starts_truth, stops_truth = get_gait_classification_truth(20.0)

        assert allclose(starts, starts_truth)
        assert allclose(stops, stops_truth)

    def test_pred_size_error(self, get_sample_accel):
        _, fs, accel = get_sample_accel(50.0)
        with pytest.raises(DimensionMismatchError):
            get_gait_classification_lgbm(random.rand(50) > 0.5, accel, fs)

    @pytest.mark.parametrize('pred', (True, False, 1, -135098135, 1.513e-600))
    def test_pred_single_input(self, pred, get_sample_accel):
        starts, stops = get_gait_classification_lgbm(pred, random.rand(500), 32.125)

        assert starts.size == 1
        assert starts[0] == 0
        assert stops.size == 1
        assert stops[0] == 500

    def test_pred_array_input(self, get_sample_accel):
        t, fs, accel = get_sample_accel(50.0)
        pred = zeros(accel.shape[0], dtype="bool")

        starts_truth = array([0, 500, 750, 950])
        stops_truth = array([150, 575, 850, 1200])
        for s, f in zip(starts_truth, stops_truth):
            pred[s:f] = True

        starts, stops = get_gait_classification_lgbm(pred, accel, fs)

        assert allclose(starts, starts_truth)
        assert allclose(stops, stops_truth)


class TestGetGaitBouts:
    @pytest.mark.parametrize('case', (1, 2, 3, 4))
    def test(self, get_bgait_samples_truth, case):
        starts, stops, time, max_sep, min_time, bouts = get_bgait_samples_truth(case)

        pred_bouts = get_gait_bouts(starts, stops, 0, 1500, time, max_sep, min_time)

        # assert allclose(pred_bouts, bouts)
        assert len(pred_bouts) == len(bouts)
        assert all([b == bouts[i] for i, b in enumerate(pred_bouts)])


class TestGetGaitEvents:
    @pytest.mark.parametrize('sign', (1, -1))
    def test_50hz(self, sign, get_sample_bout_accel, get_contact_truth):
        accel, time, axis, acc_sign = get_sample_bout_accel(50.0)
        ic_truth, fc_truth = get_contact_truth(50.0)  # index starts at 1 for this

        o_scale = round(0.4 / (2 * 1.25 / 50.0)) - 1

        ic, fc, _ = get_gait_events(
            sign * accel[:, axis],
            50.0,
            time,
            sign * acc_sign,
            o_scale, 4, 20.0, True
        )

        assert allclose(ic, ic_truth)
        assert allclose(fc, fc_truth)

    @pytest.mark.parametrize('sign', (1, -1))
    def test_20hz(self, sign, get_sample_bout_accel, get_contact_truth):
        accel, time, axis, acc_sign = get_sample_bout_accel(20)
        ic_truth, fc_truth = get_contact_truth(20)  # index starts at 1 for this

        o_scale = round(0.4 / (2 * 1.25 / 20.0)) - 1

        ic, fc, _ = get_gait_events(
            sign * accel[:, axis],
            20.0,
            time,
            sign * acc_sign,
            o_scale, 4, 20.0, False  # also test original scale
        )

        assert allclose(ic, ic_truth)
        assert allclose(fc, fc_truth)


class TestGetGaitStrides:
    def test_50hz(self, get_sample_bout_accel, get_contact_truth, get_strides_truth):
        accel, time, axis, acc_sign = get_sample_bout_accel(50.0)
        ic, fc = get_contact_truth(50.0)

        keys = ['IC', 'FC', 'FC opp foot', 'valid cycle', 'delta h']
        gait_truth = get_strides_truth(50.0, keys)

        gait = {i: [] for i in keys}
        bout_steps = get_strides(gait, accel[:, axis], 0, ic, fc, time, 50.0, 2.25, 0.2)

        assert bout_steps == 45
        for k in keys:
            assert allclose(gait[k], gait_truth[k], equal_nan=True)

    def test_20hz(self, get_sample_bout_accel, get_contact_truth, get_strides_truth):
        accel, time, axis, acc_sign = get_sample_bout_accel(20.0)
        ic, fc = get_contact_truth(20.0)

        keys = ['IC', 'FC', 'FC opp foot', 'valid cycle', 'delta h']
        gait_truth = get_strides_truth(20.0, keys)

        gait = {i: [] for i in keys}
        bout_steps = get_strides(gait, accel[:, axis], 0, ic, fc, time, 20.0, 2.25, 0.2)

        assert bout_steps == 46
        for k in keys:
            assert allclose(gait[k], gait_truth[k], equal_nan=True)

    def test_short_bout(self):
        time = arange(0, 10, 1/25)  # 25hz sample
        ic = array([10, 23])
        fc = array([12, 25, 37])

        gait = {i: [] for i in ['IC', 'FC', 'FC opp foot', 'valid cycle', 'delta h']}

        bsteps = get_strides(gait, random.rand(time.size), 0, ic, fc, time, 25.0, 2.25, 0.2)

        assert bsteps == 2
        assert allclose(gait['IC'], ic)
        assert allclose(gait['FC'], fc[1:])
        assert allclose(gait['FC opp foot'], fc[:-1])
        assert all([not i for i in gait['valid cycle']])
        assert all([isnan(i) for i in gait['delta h']])


class TestGait(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # override necessary attributes
        cls.sample_data_file = resolve_data_path('gait_data.h5', 'gait')
        cls.truth_data_file = resolve_data_path('gait_data.h5', 'gait')
        cls.truth_suffix = None
        cls.truth_data_keys = [
            'delta h',
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

    def test_leg_length_warning(self, get_sample_data, caplog):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )
        data['height'] = None

        with pytest.warns(UserWarning):
            self.process.predict(**data)

    def test_sample_rate_error(self, get_sample_data):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )
        data['time'] = arange(0, 300, 0.5)

        with pytest.raises(LowFrequencyError):
            self.process.predict(**data)

    def test_gait_predictions_error(self, get_sample_data):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )
        data['gait_pred'] = arange(0, 1, 0.1)

        with pytest.raises(ValueError):
            self.process.predict(**data)

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
