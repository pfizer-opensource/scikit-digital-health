import pytest
from numpy import asfortranarray

from skdh.utility.windowing import (
    DimensionError,
    ContiguityError,
    compute_window_samples,
    get_windowed_view,
)


class TestComputeWindowSamples:
    @pytest.mark.parametrize(
        ("fs", "w_length", "w_step", "length", "step"),
        (
            (50.0, 5.0, 1, 250, 1),
            (50.0, 5.0, 1.0, 250, 250),
            (50.0, 5.0, 0.5, 250, 125),
            (50.0, 5.0, 25, 250, 25),
        ),
    )
    def test(self, fs, w_length, w_step, length, step):
        pred_length, pred_step = compute_window_samples(fs, w_length, w_step)

        assert pred_length == length
        assert pred_step == step

    def test_float_range_error(self):
        with pytest.raises(ValueError):
            compute_window_samples(50.0, 5.0, 1.5)
        with pytest.raises(ValueError):
            compute_window_samples(50.0, 5.0, -1.5)

    @pytest.mark.parametrize(
        ("w_length", "w_step"),
        (
            (None, None),
            (None, 10),
            (5.0, None),
        ),
    )
    def test_none(self, w_length, w_step):
        pl, ps = compute_window_samples(50.0, w_length, w_step)

        assert pl is None
        assert ps is None

    def test_negative_window_step_error(self):
        with pytest.raises(ValueError):
            compute_window_samples(50.0, 5.0, -1)


class TestGetWindowedView:
    @pytest.mark.parametrize(
        ("w_length", "w_step", "res_shape"),
        (
            (250, 1, (751, 250)),
            (250, 250, (4, 250)),
        ),
    )
    def test_1d(self, w_length, w_step, res_shape, np_rng):
        # make sure that the function is broadcasting to c-array if the option
        # is set
        x = asfortranarray(np_rng.random((1000,)))

        xw = get_windowed_view(x, w_length, w_step, ensure_c_contiguity=True)

        assert xw.shape == res_shape

    @pytest.mark.parametrize(
        ("w_length", "w_step", "res_shape"),
        (
            (250, 1, (751, 250, 2)),
            (250, 250, (4, 250, 2)),
        ),
    )
    def test_2d(self, w_length, w_step, res_shape, np_rng):
        x = np_rng.random((1000, 2))

        xw = get_windowed_view(x, w_length, w_step, ensure_c_contiguity=False)

        assert xw.shape == res_shape

    def test_dim_error(self, np_rng):
        x = np_rng.random((4, 2, 2))

        with pytest.raises(DimensionError):
            get_windowed_view(x, 2, 1, ensure_c_contiguity=False)

    def test_continuous_error(self, np_rng):
        x = asfortranarray(np_rng.random((500, 2)))

        with pytest.raises(ContiguityError):
            get_windowed_view(x, 10, 1, ensure_c_contiguity=False)
