import pytest
from numpy import allclose, ndarray

from skimu.read import ReadGT3X


class TestReadGt3x:
    def test_gt3x(self, gt3x_file, gt3x_truth):
        res = ReadGT3X(base=9, period=2).predict(gt3x_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - gt3x_truth["time"][0],
            gt3x_truth["time"] - gt3x_truth["time"][0],
            atol=5e-5
        )

        assert allclose(res["accel"], gt3x_truth["accel"], atol=5e-5)

        assert all([i in res["day_ends"] for i in gt3x_truth["day_ends"]])
        assert allclose(res["day_ends"][(9, 2)], gt3x_truth['day_ends'][(9, 2)])
