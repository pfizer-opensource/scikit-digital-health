import pytest

from skdh.read import ReadApdmH5


class TestReadBin:
    def _test(self, apdm_file):
        res = ReadApdmH5("Lumbar").predict(apdm_file)

        # TODO fill this out when an actual file for testing is recieved.

    def test_none_file_error(self):
        with pytest.raises(ValueError):
            ReadApdmH5("Lumbar").predict(None)

    def test_extension_warning(self):
        with pytest.warns(UserWarning) as record:
            with pytest.raises(Exception):
                ReadApdmH5("Lumbar").predict("test.random")

        assert len(record) == 1
        assert "File extension is not expected '.h5'" in record[0].message.args[0]
