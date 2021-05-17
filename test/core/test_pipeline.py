import json
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

from skimu.pipeline import Pipeline, NotAProcessError, ProcessNotFoundError
from skimu.base import _BaseProcess
from skimu.gait import Gait


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

    def test_add(self):
        p = Pipeline()

        tp = TestProcess(kw1=500)
        p.add(tp, save_results=True, save_name="test_saver.csv")

        assert p._steps == [TestProcess(kw1=500)]
        assert tp._in_pipeline
        assert tp.pipe_save
        assert tp.pipe_fname == "test_saver.csv"

        with pytest.raises(NotAProcessError):
            p.add(list())

    def test_save(self):
        p = Pipeline()

        tp = TestProcess(kw1=2)
        # overwrite this for saving
        tp.__class__.__module__ = "skimu.test.testmodule"

        p.add(tp)

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.json"

            p.save(str(fname))
            with fname.open() as f:
                res = json.load(f)

        exp = [{
            "TestProcess": {
                "module": "test.testmodule",
                "Parameters": {"kw1": 2},
                "save_result": False,
                "save_name": "{date}_{name}_results.csv",
                "plot_save_name": None
            }
        }]

        assert res == exp

    def test_load(self):
        exp = [
            {
                "Gait": {
                    "module": "gait.gait",
                    "Parameters": {},
                    "save_result": False,
                    "save_name": "gait_results.csv",
                    "plot_save_name": None
                }
            },
            {
                "TestProcess": {
                    "module": "test.testmodule",
                    "Parameters": {},
                    "save_result": False,
                    "save_name": "test_save.csv",
                    "plot_save_name": None
                }
            }
        ]

        p = Pipeline()

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.json"

            with fname.open(mode="w") as f:
                json.dump(exp, f)

            with pytest.warns(UserWarning):
                p.load(str(fname))

        assert p._steps == [Gait()]
