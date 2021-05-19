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

    def test_window_inputs(self):
        r = ReadGT3X(base=None, period=None)
        assert not r.window

        with pytest.warns(UserWarning) as record:
            r = ReadGT3X(base=8, period=None)
            r = ReadGT3X(base=None, period=12)

        assert len(record) == 2
        assert "One of base or period is None" in record[0].message.args[0]
        assert "One of base or period is None" in record[1].message.args[0]

    def test_window_range_error(self):
        with pytest.raises(ValueError):
            ReadGT3X(base=24, period=26)

    def test_none_file_error(self):
        with pytest.raises(ValueError):
            ReadGT3X().predict(None)

    def test_extension_warning(self):
        with pytest.warns(UserWarning) as record:
            with pytest.raises(Exception):
                ReadGT3X().predict("test.random")

        assert len(record) == 1
        assert "File extension is not expected '.gt3x'" in record[0].message.args[0]
