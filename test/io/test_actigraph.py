from tempfile import NamedTemporaryFile

import pytest
from numpy import allclose

from skdh.io import ReadGT3X, FileSizeError


class TestReadGt3x:
    def test_gt3x(self, gt3x_file, gt3x_truth):
        res = ReadGT3X(base=9, period=2).predict(gt3x_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - gt3x_truth["time"][0],
            gt3x_truth["time"] - gt3x_truth["time"][0],
            atol=5e-5,
        )

        assert allclose(res["accel"], gt3x_truth["accel"], atol=5e-5)

        assert all([i in res["day_ends"] for i in gt3x_truth["day_ends"]])
        assert allclose(res["day_ends"][(9, 2)], gt3x_truth["day_ends"][(9, 2)])

    def test_window_inputs(self):
        r = ReadGT3X(base=None, period=None)
        assert not r.window

        with pytest.warns(UserWarning, match="One of base or period is None"):
            r = ReadGT3X(base=8, period=None)
            r = ReadGT3X(base=None, period=12)

    def test_window_range_error(self):
        with pytest.raises(ValueError):
            ReadGT3X(base=24, period=26)

    def test_extension(self):
        with NamedTemporaryFile(suffix=".abc") as tmpf:
            with pytest.warns(UserWarning, match=r"expected \[.gt3x\]"):
                with pytest.raises(Exception):
                    ReadGT3X().predict(tmpf.name)

    def test_small_size(self):
        ntf = NamedTemporaryFile(mode="w", suffix=".gt3x")

        ntf.writelines(["a\n", "b\n", "c\n"])

        with pytest.raises(FileSizeError):
            ReadGT3X().predict(ntf.name)

        ntf.close()
