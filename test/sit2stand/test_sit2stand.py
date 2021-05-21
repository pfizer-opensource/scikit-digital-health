import pytest
from numpy import isclose, allclose

from skimu.sit2stand import Sit2Stand


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

    def test(self):
        pass
