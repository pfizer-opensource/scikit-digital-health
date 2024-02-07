import pytest
from numpy import isclose, allclose

from skdh.sit2stand import Sit2Stand


class TestSit2Stand:
    def test_power_band(self):
        s2s = Sit2Stand(power_band=0.5)

        assert isclose(s2s.power_start_f, 0.0)
        assert isclose(s2s.power_end_f, 0.5)

        s2s = Sit2Stand(power_band=(0.5, 0.8))
        assert isclose(s2s.power_start_f, 0.5)
        assert isclose(s2s.power_end_f, 0.8)

    def test_power_peak_kw(self):
        s2s = Sit2Stand(power_peak_kw=None)
        assert s2s.power_peak_kw == {"height": 90 / 9.81}

        s2s = Sit2Stand(power_peak_kw={"height": 5})
        assert s2s.power_peak_kw == {"height": 5}

    def test_stillness(self, s2s_input, stillness_truth):
        s2s = Sit2Stand(
            stillness_constraint=True,
            gravity=9.81,
            thresholds=None,
            long_still=0.5,
            still_window=0.3,
            gravity_pass_order=4,
            gravity_pass_cutoff=0.8,
            continuous_wavelet="gaus1",
            power_band=[0, 0.5],
            power_peak_kw={"distance": 128},
            power_std_height=True,
            power_std_trim=0,
            lowpass_order=4,
            lowpass_cutoff=5,
            reconstruction_window=0.25,
        )

        res = s2s.predict(time=s2s_input["time"], accel=s2s_input["accel"])

        for k in stillness_truth:
            assert allclose(res[k], stillness_truth[k])

    def test_displacement(self, s2s_input, displacement_truth):
        s2s = Sit2Stand(
            stillness_constraint=False,
            gravity=9.81,
            thresholds=None,
            long_still=0.5,
            still_window=0.3,
            gravity_pass_order=4,
            gravity_pass_cutoff=0.8,
            continuous_wavelet="gaus1",
            power_band=[0, 0.5],
            power_peak_kw={"distance": 128},
            power_std_height=True,
            power_std_trim=0,
            lowpass_order=4,
            lowpass_cutoff=5,
            reconstruction_window=0.25,
        )

        res = s2s.predict(time=s2s_input["time"], accel=s2s_input["accel"])

        for k in displacement_truth:
            assert allclose(res[k], displacement_truth[k])
