from tempfile import NamedTemporaryFile

import pytest
from numpy import allclose
from pandas import to_datetime

from skdh.io import ReadCwa
from skdh.utility.exceptions import FileSizeError


class TestReadCwa:
    def test_ax3(self, ax3_file, ax3_truth):
        res = ReadCwa().predict(file=ax3_file)

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

    def test_ax6(self, ax6_file, ax6_truth):
        res = ReadCwa().predict(file=ax6_file)

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

    def test_ax6_tz(self, ax6_file, ax6_truth):
        res = ReadCwa().predict(file=ax6_file, tz_name="US/Eastern")

        # adjust the truth timestamps
        truth_time_ = to_datetime(ax6_truth["time"], unit="s", utc=False).tz_localize(
            "US/Eastern"
        )
        truth_time = truth_time_.view("int64") / 1e9

        assert allclose(
            res["time"] - truth_time[0], truth_time - truth_time[0], atol=5e-5
        )

        for k in ["accel", "gyro", "temperature", "fs"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], ax6_truth[k], atol=5e-5)

    def test_extension(self):
        with NamedTemporaryFile(suffix=".abc") as tmpf:
            with pytest.warns(UserWarning, match=r"expected \[.cwa\]"):
                with pytest.raises(Exception):
                    ReadCwa().predict(file=tmpf.name)

    def test_small_size(self):
        ntf = NamedTemporaryFile(mode="w", suffix=".cwa")

        ntf.writelines(["a\n", "b\n", "c\n"])

        with pytest.raises(FileSizeError):
            ReadCwa().predict(file=ntf.name)

        ntf.close()
