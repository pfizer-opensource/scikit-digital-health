import json
from tempfile import TemporaryDirectory
from pathlib import Path

from skimu.pipeline import Pipeline, NotAProcessError, ProcessNotFoundError
from skimu.base import _BaseProcess


# temp class just for testing
class TestProcess(_BaseProcess):
    def __init__(self, kw1=1):
        super().__init__(kw1=kw1)


class TestPipeline:
    def test_str_repr(self):
        p = Pipeline()

        assert repr(p) == "IMUAnalysisPipeline[\n]"

        p.add(TestProcess(kw1=2))
        p.add(TestProcess(kw1=1))

        assert str(p) == "IMUAnalysisPipeline"
        assert repr(p) == "IMUAnalysisPipeline[\n\tTestProcess(kw1=2),\n\tTestProcess(kw1=1),\n]"

    def test_save(self):
        p = Pipeline()

        tp = TestProcess(kw1=2)
        # overwrite this for saving
        tp.__class__.__module__ = "skimu.test.testmodule"

        p.add(tp)

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.json"

            p.save(str(fname))
            with open(str(fname)) as f:
                res = json.load(f)

        exp = [{
            "TestProcess": {
                "module": "test.testmodule",
                "Parameters": {"kw": 2},
                "save_result": False,
                "save_name": "{date}_{name}_results.csv",
                "plot_save_name": None
            }
        }]

        assert res == exp
