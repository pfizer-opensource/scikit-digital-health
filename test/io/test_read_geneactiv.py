from tempfile import NamedTemporaryFile

import pytest
from numpy import allclose, ndarray

from skdh.io import ReadBin, FileSizeError


class TestReadBin:
    def test(self, gnactv_file, gnactv_truth):
        res = ReadBin(bases=8, periods=12).predict(gnactv_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - gnactv_truth["time"][0],
            gnactv_truth["time"] - gnactv_truth["time"][0],
        )

        for k in ["accel", "temperature", "light"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], gnactv_truth[k], atol=5e-5)

        assert all([i in res["day_ends"] for i in gnactv_truth["day_ends"]])
        assert allclose(res["day_ends"][(8, 12)], gnactv_truth["day_ends"][(8, 12)])

    def test_window_inputs(self):
        r = ReadBin(bases=None, periods=None)
        assert not r.window
        assert isinstance(r.bases, ndarray)
        assert isinstance(r.periods, ndarray)

        with pytest.warns(UserWarning) as record:
            r = ReadBin(bases=8, periods=None)
            r = ReadBin(bases=None, periods=12)

        assert len(record) == 2
        assert "One of base or period is None" in record[0].message.args[0]
        assert "One of base or period is None" in record[1].message.args[0]

    def test_window_range_error(self):
        with pytest.raises(ValueError):
            ReadBin(bases=[0, 24], periods=[5, 26])

    def test_extension(self):
        with NamedTemporaryFile(suffix=".abc") as tmpf:
            with pytest.warns(UserWarning, match=r"expected \[.bin\]"):
                with pytest.raises(FileSizeError):
                    ReadBin().predict(tmpf.name)

    def test_small_size(self):
        ntf = NamedTemporaryFile(mode="w", suffix=".bin")

        ntf.writelines(["a\n", "b\n", "c\n"])

        with pytest.raises(FileSizeError):
            ReadBin().predict(ntf.name)

        ntf.close()
