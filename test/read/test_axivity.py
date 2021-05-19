import pytest
from numpy import allclose, ndarray

from skimu.read import ReadCWA


class TestReadBin:
    def test(self, ax3_file, ax3_truth):
        res = ReadCWA(bases=8, periods=12).predict(ax3_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - ax3_truth["time"][0],
            ax3_truth["time"] - ax3_truth["time"][0],
            atol=5e-5
        )

        for k in ["accel", "temperature", "fs"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], ax3_truth[k], atol=5e-5)

        assert all([i in res["day_ends"] for i in ax3_truth["day_ends"]])
        assert allclose(res["day_ends"][(8, 12)], ax3_truth['day_ends'][(8, 12)])

