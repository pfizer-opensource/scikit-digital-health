import pytest
from numpy import allclose, array

from skdh.gait.gait import Gait, LowFrequencyError
from skdh.gait import gait_endpoints


class TestGait:
    def test(self, gait_input_50, gait_res_50):
        t, acc = gait_input_50

        g = Gait(
            correct_accel_orient=True,
            use_cwt_scale_relation=True,
            min_bout_time=8.0,
            max_bout_separation_time=0.25,
            max_stride_time=2.25,
            loading_factor=0.2,
            height_factor=0.53,
            prov_leg_length=False,
            filter_order=4,
            filter_cutoff=20.0,
            downsample_aa_filter=True,
            day_window=(0, 24),
        )

        res = g.predict(time=t, accel=acc, fs=50.0, height=1.88)

        for key in gait_res_50.files:
            assert allclose(res[key], gait_res_50[key], equal_nan=True), key

    def test_with_turns(self, gait_input_gyro, gait_res_gyro):
        t, acc, gyr = gait_input_gyro

        g = Gait(
            correct_accel_orient=True,
            use_cwt_scale_relation=True,
            min_bout_time=8.0,
            max_bout_separation_time=0.25,
            max_stride_time=2.25,
            loading_factor=0.2,
            height_factor=0.53,
            prov_leg_length=False,
            filter_order=4,
            filter_cutoff=20.0,
            downsample_aa_filter=True,
            day_window=(0, 24),
        )

        res = g.predict(
            time=t, accel=acc, gyro=gyr, fs=128.0, height=1.88, gait_pred=True
        )

        for key in gait_res_gyro.files:
            assert allclose(res[key], gait_res_gyro[key], equal_nan=True), key

    def test_add_metrics(self):
        g = Gait()
        g._params = []  # reset for easy testing

        g.add_endpoints([gait_endpoints.StrideTime, gait_endpoints.StepTime])
        g.add_endpoints(gait_endpoints.PhaseCoordinationIndex)

        assert g._params == [
            gait_endpoints.StrideTime,
            gait_endpoints.StepTime,
            gait_endpoints.PhaseCoordinationIndex,
        ]

    def test_add_metrics_error(self):
        g = Gait()

        with pytest.raises(ValueError):
            g.add_endpoints([list, Gait])

        with pytest.raises(ValueError):
            g.add_endpoints(Gait)

    def test_leg_length_factor(self):
        g = Gait(prov_leg_length=True, height_factor=0.53)

        assert g.height_factor == 1.0

    def test__handle_input_gait_predictions(self):
        starts, stops = Gait._handle_input_gait_predictions(None, 10)
        assert starts is None and stops is None

        starts, stops = Gait._handle_input_gait_predictions(True, 10)
        assert allclose(starts, [0])
        assert allclose(stops, [9])

        starts, stops = Gait._handle_input_gait_predictions(
            array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0]), 12
        )
        assert allclose(starts, [3, 10])
        assert allclose(stops, [9, 11])

        with pytest.raises(ValueError):
            Gait._handle_input_gait_predictions(array([1, 1, 1, 0, 0, 1]), 15)

    def test_no_height_low_frequency(self):
        t = array([1, 2, 3])
        a = array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(LowFrequencyError):
            with pytest.warns(UserWarning):
                Gait().predict(time=t, accel=a, fs=10.0, height=None)
