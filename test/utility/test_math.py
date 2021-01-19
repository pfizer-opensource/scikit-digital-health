import pytest

import numpy as np

from skimu.utility import get_windowed_view
from skimu.utility.math import *


class TestRollingMean:
    def test(self):
        x = np.random.random(10000)
        xw = get_windowed_view(x, 150, 150)

        truth = np.mean(xw, axis=-1)
        pred = rolling_mean(x, 150, 150)

        assert np.allclose(pred, truth)

    @pytest.mark.parametrize(
        ("in_shape", "out_shape", "kwargs"),
        (
                ((5, 500), (5, 21), {"w_len": 100, "skip": 20, "axis": -1}),
                ((500, 5), (21, 5), {"w_len": 100, "skip": 20, "axis": 0}),
                ((500,), (21,), {"w_len": 100, "skip": 20}),
                ((3, 10, 3187), (3, 10, 3015), {"w_len": 173, "skip": 1, "axis": -1}),
        )
    )
    def test_in_out_shapes(self, in_shape, out_shape, kwargs):
        x = np.random.random(in_shape)
        pred = rolling_mean(x, **kwargs)

        assert pred.shape == out_shape

    def test_window_length_shape_error(self):
        x = np.random.random((5, 10))

        with pytest.raises(ValueError):
            rolling_mean(x, 11, 1, axis=-1)

    def test_negative_error(self):
        x = np.random.random((100, 300))
        for args in (-1, 10), (10, -1), (-5, -5):
            with pytest.raises(ValueError):
                rolling_mean(x, *args, axis=-1)
