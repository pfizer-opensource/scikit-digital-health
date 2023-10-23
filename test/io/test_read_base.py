from tempfile import NamedTemporaryFile

import pytest

from skdh import Pipeline


def test_in_pipeline(dummy_reader_class, dummy_process):
    rdr = dummy_reader_class()
    proc = dummy_process()

    p = Pipeline()
    p.add(rdr)
    p.add(proc)

    with NamedTemporaryFile(suffix=".abc") as tmpf:
        res = p.run(file=tmpf.name)

    assert res == {
        "Rdr": {"in_predict": True, "test_input": 5},
        "dummyprocess": {"a": 5, "test_output": 10.0},
    }


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
        rdr._in_pipeline = True
        # due to self in the predict method, dummy predict must be wrapped in conftest

        with NamedTemporaryFile(suffix=".abc") as tmpf:
            kw, res = rdr.predict(file=tmpf.name, testkw="testkw")

            assert kw["file"] == tmpf.name
            assert kw["testkw"] == "testkw"
            # make sure the value in predict IS set
            assert "in_predict" in res
            assert res["in_predict"]
