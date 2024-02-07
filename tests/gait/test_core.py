import pytest
from numpy import allclose, array

from skdh import Pipeline
from skdh.gait.core import GaitLumbar
from skdh.gait import substeps
from skdh.gait import gait_metrics
from skdh.context import PredictGaitLumbarLgbm
from skdh.utility.exceptions import LowFrequencyError


class TestGait:
    def test_apcwt(self, gait_input_50, gait_res_50_apcwt):
        t, acc = gait_input_50

        g = GaitLumbar(
            downsample=False,
            height_factor=0.53,
            provide_leg_length=False,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            gait_event_method="AP CWT",
            correct_orientation=True,
            filter_cutoff=20.0,
            filter_order=4,
            ic_prom_factor=0.1,  # match original processing for testing
            ic_dist_factor=0.0,
            fc_prom_factor=0.1,
            fc_dist_factor=0.0,
            max_stride_time=lambda x: 2.0 * x + 1.0,
            loading_factor=lambda x: 0.17 * x + 0.05,
        )

        cls = PredictGaitLumbarLgbm()
        cls._in_pipeline = True
        kw, bouts = cls.predict(time=t, accel=acc, fs=50.0)

        res = g.predict(height=1.88, **kw)

        for key in gait_res_50_apcwt:
            assert allclose(res[key], gait_res_50_apcwt[key], equal_nan=True), key

    def test_vcwt(self, gait_input_50, gait_res_50_vcwt):
        t, acc = gait_input_50

        g = GaitLumbar(
            downsample=True,
            height_factor=0.53,
            provide_leg_length=False,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            gait_event_method="v cwt",
            correct_orientation=True,
            filter_cutoff=20.0,
            filter_order=4,
            use_cwt_scale_relation=True,
            wavelet_scale="default",
            round_scale=True,
            max_stride_time=2.25,
            loading_factor=0.2,
        )

        cls = PredictGaitLumbarLgbm()
        cls._in_pipeline = True
        kw, bouts = cls.predict(time=t, accel=acc, fs=50.0)

        res = g.predict(height=1.88, **kw)

        for key in gait_res_50_vcwt.files:
            assert allclose(res[key], gait_res_50_vcwt[key], equal_nan=True), key

    def test_with_turns_apcwt(self, gait_input_gyro, gait_res_gyro_apcwt):
        t, acc, gyr = gait_input_gyro

        g = GaitLumbar(
            downsample=False,
            height_factor=0.53,
            provide_leg_length=False,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            gait_event_method="AP CWT",
            correct_orientation=True,
            filter_cutoff=20.0,
            filter_order=4,
            ic_prom_factor=0.1,  # match original processing for testing
            ic_dist_factor=0.0,
            fc_prom_factor=0.1,
            fc_dist_factor=0.0,
            max_stride_time=lambda x: 2.0 * x + 1.0,
            loading_factor=lambda x: 0.17 * x + 0.05,
        )

        res = g.predict(
            time=t, accel=acc, gyro=gyr, fs=128.0, height=1.88, gait_pred=True
        )

        for key in gait_res_gyro_apcwt:
            assert allclose(res[key], gait_res_gyro_apcwt[key], equal_nan=True), key

    def test_with_turns_vcwt(self, gait_input_gyro, gait_res_gyro_vcwt):
        t, acc, gyr = gait_input_gyro

        g = GaitLumbar(
            downsample=True,
            height_factor=0.53,
            provide_leg_length=False,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            gait_event_method="v cwt",
            correct_orientation=True,
            filter_cutoff=20.0,
            filter_order=4,
            use_cwt_scale_relation=False,
            wavelet_scale=8,  # hard-code scale so it matches original test
            round_scale=True,
            max_stride_time=2.25,
            loading_factor=0.2,
        )

        res = g.predict(
            time=t, accel=acc, gyro=gyr, fs=128.0, height=1.88, gait_pred=True
        )

        for key in gait_res_gyro_vcwt.files:
            assert allclose(res[key], gait_res_gyro_vcwt[key], equal_nan=True), key

    def test_event_method_input_error(self):
        with pytest.raises(ValueError):
            g = GaitLumbar(gait_event_method="test")

    def test_bout_pipeline_input(self):
        b = Pipeline()
        b.add(substeps.PreprocessGaitBout())

        g = GaitLumbar(bout_processing_pipeline=b)

        assert g.bout_pipeline == b

        with pytest.raises(ValueError):
            g = GaitLumbar(bout_processing_pipeline="test")

    def test_add_metrics(self):
        g = GaitLumbar()
        g._params = []  # reset for easy testing

        g.add_endpoints([gait_metrics.StrideTime, gait_metrics.StepTime])
        g.add_endpoints(gait_metrics.PhaseCoordinationIndex)

        assert g._params == [
            gait_metrics.StrideTime,
            gait_metrics.StepTime,
            gait_metrics.PhaseCoordinationIndex,
        ]

    def test_add_metrics_error(self):
        g = GaitLumbar()

        with pytest.raises(ValueError):
            g.add_endpoints([list, GaitLumbar])

        with pytest.raises(ValueError):
            g.add_endpoints(GaitLumbar)

    def test_leg_length_factor(self):
        g = GaitLumbar(provide_leg_length=True, height_factor=0.53)

        assert g.height_factor == 1.0

    def test__handle_input_gait_predictions(self):
        # ====================================
        # gait_pred values
        starts, stops = GaitLumbar._handle_input_gait_predictions(None, True, 10)
        assert allclose(starts, [0])
        assert allclose(stops, [9])

        starts, stops = GaitLumbar._handle_input_gait_predictions(
            None, array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0]), 12
        )
        assert allclose(starts, [3, 10])
        assert allclose(stops, [9, 11])

        with pytest.raises(ValueError):
            GaitLumbar._handle_input_gait_predictions(
                None, array([1, 1, 1, 0, 0, 1]), 15
            )

        # ====================================
        # gait_bout values
        starts, stops = GaitLumbar._handle_input_gait_predictions(
            array([[0, 4], [6, 8]]), None, 10
        )
        assert allclose(starts, [0, 6])
        assert allclose(stops, [4, 8])

        starts, stops = GaitLumbar._handle_input_gait_predictions(
            array([[0, 4], [6, 8]]), True, 10
        )
        assert allclose(starts, [0, 6])
        assert allclose(stops, [4, 8])

    def test_no_height_low_frequency(self):
        t = array([1, 2, 3])
        a = array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(LowFrequencyError):
            with pytest.warns(UserWarning):
                GaitLumbar().predict(time=t, accel=a, fs=10.0, height=None)
