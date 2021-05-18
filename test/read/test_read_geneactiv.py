import pytest
from numpy import allclose

from skimu.read import ReadBin


class TestReadBin:
    def test(self, gnactv_file, gnactv_truth):
        res = ReadBin(bases=8, periods=12).predict(gnactv_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - gnactv_truth["time"][0],
            gnactv_truth["time"] - gnactv_truth["time"][0]
        )

        for k in ["accel", "temperature", "light"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], gnactv_truth[k], atol=5e-5)

        assert res["day_ends"] == gnactv_truth["day_ends"]

    def test_none_file_error(self):
        with pytest.raises(ValueError):
            ReadBin().predict(None)

    def test_extension_warning(self):
        with pytest.warns(UserWarning) as record:
            with pytest.raises(Exception):
                ReadBin().predict("test.random")

        assert len(record) == 1
        assert "File extension is not expected '.bin'" in record[0].message.args[0]
