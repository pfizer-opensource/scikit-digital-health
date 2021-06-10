import pytest
from numpy import allclose, ndarray

from skdh.read import ReadApdmH5


class TestReadBin:
    def test(self, apdm_file, apdm_truth):
        res = ReadApdmH5("Lumbar").predict(apdm_file)

        # make sure it will catch small differences
        assert allclose(
            res["time"] - apdm_truth["time"][0],
            apdm_truth["time"] - apdm_truth["time"][0],
        )

        for k in ["accel", "temperature", "light"]:
            # adjust tolerance - GeneActiv truth values from the CSV
            # were truncated by rounding
            assert allclose(res[k], apdm_truth[k], atol=5e-5)

        assert all([i in res["day_ends"] for i in apdm_truth["day_ends"]])
        assert allclose(res["day_ends"][(8, 12)], apdm_truth["day_ends"][(8, 12)])

    def test_none_file_error(self):
        with pytest.raises(ValueError):
            ReadApdmH5("Lumbar").predict(None)

    def test_extension_warning(self):
        with pytest.warns(UserWarning) as record:
            with pytest.raises(Exception):
                ReadApdmH5("Lumbar").predict("test.random")

        assert len(record) == 1
        assert "File extension is not expected '.h5'" in record[0].message.args[0]
