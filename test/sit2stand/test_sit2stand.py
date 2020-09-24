from ..base_conftest import BaseProcessTester

from PfyMU.sit2stand import Sit2Stand


class TestSit2StandStillness(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # override specific necessary attributes
        cls.sample_data_file = ('PfyMU.sit2stand.tests.data', 'test_data.h5')
        cls.truth_data_file = ('PfyMU.sit2stand.tests.data', 'test_data.h5')
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
        cls.sample_data_file = ('PfyMU.sit2stand.tests.data', 'test_data.h5')
        cls.truth_data_file = ('PfyMU.sit2stand.tests.data', 'test_data.h5')
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
