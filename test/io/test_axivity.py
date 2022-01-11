from tempfile import NamedTemporaryFile

import pytest
from numpy import allclose, ndarray

from skdh.io import ReadCwa, FileSizeError


class TestReadCwa:
    def test_ax3(self, ax3_file, ax3_truth):
        res = ReadCwa(bases=8, periods=12).predict(ax3_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - ax3_truth["time"][0],
            ax3_truth["time"] - ax3_truth["time"][0],
            atol=5e-5,
        )

        for k in ["accel", "temperature", "fs"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], ax3_truth[k], atol=5e-5)

        assert all([i in res["day_ends"] for i in ax3_truth["day_ends"]])
        assert allclose(res["day_ends"][(8, 12)], ax3_truth["day_ends"][(8, 12)])

    def test_ax6(self, ax6_file, ax6_truth):
        res = ReadCwa(bases=8, periods=12).predict(ax6_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - ax6_truth["time"][0],
            ax6_truth["time"] - ax6_truth["time"][0],
            atol=5e-5,
        )

        for k in ["accel", "gyro", "temperature", "fs"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], ax6_truth[k], atol=5e-5)

        assert all([i in res["day_ends"] for i in ax6_truth["day_ends"]])
        assert allclose(res["day_ends"][(8, 12)], ax6_truth["day_ends"][(8, 12)])

    def test_window_inputs(self):
        r = ReadCwa(bases=None, periods=None)
        assert not r.window
        assert isinstance(r.bases, ndarray)
        assert isinstance(r.periods, ndarray)

        with pytest.warns(UserWarning) as record:
            r = ReadCwa(bases=8, periods=None)
            r = ReadCwa(bases=None, periods=12)

        assert len(record) == 2
        assert "One of base or period is None" in record[0].message.args[0]
        assert "One of base or period is None" in record[1].message.args[0]

    def test_window_range_error(self):
        with pytest.raises(ValueError):
            ReadCwa(bases=[0, 24], periods=[5, 26])

    def test_extension(self):
        with NamedTemporaryFile(suffix=".abc") as tmpf:
            with pytest.warns(UserWarning, match=r"expected \[.cwa\]"):
                with pytest.raises(Exception):
                    ReadCwa().predict(tmpf.name)

    def test_small_size(self):
        ntf = NamedTemporaryFile(mode="w", suffix=".cwa")

        ntf.writelines(["a\n", "b\n", "c\n"])

        with pytest.raises(FileSizeError):
            ReadCwa().predict(ntf.name)

        ntf.close()
