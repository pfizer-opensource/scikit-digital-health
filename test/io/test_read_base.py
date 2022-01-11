from tempfile import NamedTemporaryFile

import pytest


class TestCheckInputFile:
    def test_none(self, dummy_reader_class):
        rdr = dummy_reader_class()
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with pytest.raises(ValueError):
            rdr.predict(file=None)

    def test_not_exists(self, dummy_reader_class):
        rdr = dummy_reader_class()
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with pytest.raises(FileNotFoundError):
            rdr.predict(file="test.file")

    def test_suffix_warn(self, dummy_reader_class):
        rdr = dummy_reader_class(ext_error="warn")
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with NamedTemporaryFile(suffix=".cba") as tmpf:
            with pytest.warns(
                UserWarning, match=r"File extension \[.cba\] does not match expected"
            ):
                rdr.predict(file=tmpf.name)

    def test_suffix_raise(self, dummy_reader_class):
        rdr = dummy_reader_class(ext_error="raise")
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with NamedTemporaryFile(suffix=".cba") as tmpf:
            with pytest.raises(
                ValueError, match=r"File extension \[.cba\] does not match expected"
            ):
                rdr.predict(file=tmpf.name)

    def test_suffix_skip(self, dummy_reader_class):
        rdr = dummy_reader_class(ext_error="skip")
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with NamedTemporaryFile(suffix=".cba") as tmpf:
            kw = rdr.predict(file=tmpf.name, testkw="testkw")

            assert kw["file"] == tmpf.name
            assert kw["testkw"] == "testkw"
            # make sure the value in predict IS NOT set
            assert "in_predict" not in kw

    def test_pass_through_decorator(self, dummy_reader_class):
        rdr = dummy_reader_class(ext_error="raise")
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with NamedTemporaryFile(suffix=".abc") as tmpf:
            kw = rdr.predict(file=tmpf.name, testkw="testkw")

            assert kw["file"] == tmpf.name
            assert kw["testkw"] == "testkw"
            # make sure the value in predict IS set
            assert "in_predict" in kw
            assert kw["in_predict"]
