from tempfile import NamedTemporaryFile

import pytest
from numpy import allclose, ndarray

from skdh.io import ReadBin
from skdh.utility.exceptions import FileSizeError


class TestReadBin:
    def test(self, gnactv_file, gnactv_truth):
        res = ReadBin().predict(file=gnactv_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - gnactv_truth["time"][0],
            gnactv_truth["time"] - gnactv_truth["time"][0],
        )

        for k in ["accel", "temperature", "light"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], gnactv_truth[k], atol=5e-5)

    def test_tz(self, gnactv_file, gnactv_truth):
        # adjust the geneactive truth time to be actual UTC timestamps
        gnactv_truth["time"] = gnactv_truth["time"] + 3600 * 4

        res = ReadBin().predict(file=gnactv_file, tz_name="US/Eastern")

        # make sure it will catch small differences
        assert allclose(
            res["time"] - gnactv_truth["time"][0],
            gnactv_truth["time"] - gnactv_truth["time"][0],
        )

        for k in ["accel", "temperature", "light"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], gnactv_truth[k], atol=5e-5)

    def test_extension(self):
        with NamedTemporaryFile(suffix=".abc") as tmpf:
            with pytest.warns(UserWarning, match=r"expected \[.bin\]"):
                with pytest.raises(FileSizeError):
                    ReadBin().predict(file=tmpf.name)

    def test_small_size(self):
        ntf = NamedTemporaryFile(mode="w", suffix=".bin")

        ntf.writelines(["a\n", "b\n", "c\n"])

        with pytest.raises(FileSizeError):
            ReadBin().predict(file=ntf.name)

        ntf.close()
