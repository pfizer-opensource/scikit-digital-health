import pytest

from numpy import asfortranarray, random

from skimu.utility import get_windowed_view, compute_window_samples
from skimu.utility.windowing import DimensionError, ContiguityError


class TestGetWindowedView:
    def test_ndim_error(self):
        with pytest.raises(DimensionError):
            get_windowed_view(random.rand(5, 3, 4, 1), 5, 5)

    def test_c_cont_error(self):
        with pytest.raises(ContiguityError):
            get_windowed_view(asfortranarray(random.rand(10, 3)), 3, 3)


class TestComputeWindowSamples:
    @pytest.mark.parametrize(
        ("fs", "L", "S", "res_l", "res_s"),
        (
            (50.0, 3.0, 1.0, 150, 150),
            (50.0, 1.5, 300, 75, 300),
            (100.0, 2.5, 0.5, 250, 125),
            (10.0, 3.0, 0.0001, 30, 1),
        ),
    )
    def test(self, fs, L, S, res_l, res_s):
        nl, ns = compute_window_samples(fs, L, S)

        assert nl == res_l
        assert ns == res_s

    def test_negative_step_error(self):
        with pytest.raises(ValueError):
            compute_window_samples(50.0, 3.0, -10)

    def test_float_error(self):
        with pytest.raises(ValueError):
            compute_window_samples(50.0, 3.0, 1.1)
