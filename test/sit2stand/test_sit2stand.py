import pytest
import numpy as np

from ..base_conftest import *

from skimu.sit2stand import Sit2Stand
from skimu.sit2stand.detector import moving_stats


class TestMovingStats:
    def test(self):
        a = np.arange(1, 11)

        rm, rsd, pad = moving_stats(a, 3)

        assert np.allclose(rm, np.array([2, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
        assert np.allclose(rsd, [np.std([1, 2, 3], ddof=1)] * a.size)
        assert pad == 2

    def test4(self):
        a = np.arange(1, 11)

        rm, rsd, pad = moving_stats(a, 4)

        assert np.allclose(rm, np.array([2.5, 2.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 8.5]))
        assert np.allclose(rsd, [np.std([1, 2, 3, 4], ddof=1)] * a.size)
        assert pad == 2

    def test_window_1(self):
        a = np.arange(1, 11)
        n = a.size

        rm, rsd, pad = moving_stats(a, 1)
        del a

        assert np.allclose(rm, np.array([1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]))
        assert np.allclose(rsd, [np.std([1, 2], ddof=1)] * n)
        assert pad == 1

    def test_2d_error(self):
        a = np.random.rand(500, 2)

        with pytest.raises(ValueError):
            moving_stats(a, 17)


class TestSit2StandStillness(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # override specific necessary attributes
        cls.sample_data_file = resolve_data_path('sit2stand_data.h5', 'sit2stand')
        cls.truth_data_file = resolve_data_path('sit2stand_data.h5', 'sit2stand')
        cls.truth_suffix = 'Stillness'
        cls.truth_data_keys = [
            'STS Start',
            'STS End',
            'Duration',
            'Max. Accel.',
            'Min. Accel.',
            'SPARC',
            'Vertical Displacement'
        ]

        cls.process = Sit2Stand(
            stillness_constraint=True,
            gravity=9.81,
            thresholds=None,
            long_still=0.5,
            still_window=0.3,
            gravity_pass_order=4,
            gravity_pass_cutoff=0.8,
            continuous_wavelet='gaus1',
            power_band=[0, 0.5],
            power_peak_kw={'distance': 128},
            power_std_height=True,
            power_std_trim=0,
            lowpass_order=4,
            lowpass_cutoff=5,
            reconstruction_window=0.25
        )


class TestSit2StandDisplacement(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # override specific necessary attributes
        cls.sample_data_file = resolve_data_path('sit2stand_data.h5', 'sit2stand')
        cls.truth_data_file = resolve_data_path('sit2stand_data.h5', 'sit2stand')
        cls.truth_suffix = 'Displacement'
        cls.truth_data_keys = [
            'STS Start',
            'STS End',
            'Duration',
            'Max. Accel.',
            'Min. Accel.',
            'SPARC',
            'Vertical Displacement'
        ]

        cls.process = Sit2Stand(
            stillness_constraint=False,
            gravity=9.81,
            thresholds=None,
            long_still=0.5,
            still_window=0.3,
            gravity_pass_order=4,
            gravity_pass_cutoff=0.8,
            continuous_wavelet='gaus1',
            power_band=[0, 0.5],
            power_peak_kw={'distance': 128},
            power_std_height=True,
            power_std_trim=0,
            lowpass_order=4,
            lowpass_cutoff=5,
            reconstruction_window=0.25
        )
