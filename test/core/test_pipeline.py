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
        self.kw1 = kw1

    def predict(self, *args, **kwargs):
        self.logger.info(f"kw1={self.kw1}")

        return {"kw1": self.kw1}, {"kw1": self.kw1}


class TestProcess2(_BaseProcess):
    def __init__(self, kwa=5):
        super().__init__(kwa=kwa)
        self.kwa = kwa

    def predict(self, *args, **kwargs):
        self.logger.info(f"kwa={self.kwa}")

        return {"kwa": self.kwa}, {"kwa": self.kwa}


class TestPipeline:
    @staticmethod
    def setup_lgr():
        class Lgr:
            msgs = []

            def info(self, msg):
                self.msgs.append(msg)

        return Lgr()

    def test_run(self):
        p = Pipeline()

        tp1 = TestProcess(kw1=1)
        tp1.logger = self.setup_lgr()
        tp2 = TestProcess2(kwa=5)
        tp2.logger = self.setup_lgr()

        p.add(tp1)
        p.add(tp2)

        res = p.run()

        assert "kw1=1" in tp1.logger.msgs
        assert "kwa=5" in tp2.logger.msgs

        exp_res = {"TestProcess": {"kw1": 1}, "TestProcess2": {"kwa": 5}}

        assert res == exp_res

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
        p.add(tp, save_file="test_saver.csv")

        assert p._steps == [TestProcess(kw1=500)]
        assert tp._in_pipeline
        assert tp.pipe_save_file == "test_saver.csv"

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
                "save_file": None,
                "plot_file": None
            }
        }]

        assert res == exp

    def test_load_through_init(self):
        exp = [
            {
                "Gait": {
                    "module": "gait.gait",
                    "Parameters": {},
                    "save_file": "gait_results.csv",
                    "plot_file": None
                }
            },
            {
                "TestProcess": {
                    "module": "test.testmodule",
                    "Parameters": {},
                    "save_file": None,
                    "plot_file": None
                }
            }
        ]

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.json"

            with fname.open(mode="w") as f:
                json.dump(exp, f)

            with pytest.warns(UserWarning):
                p = Pipeline(str(fname))

        assert p._steps == [Gait()]

    def test_load_function(self):
        exp = [
            {
                "Gait": {
                    "module": "gait.gait",
                    "Parameters": {},
                    "save_file": None,
                    "plot_file": None
                }
            },
            {
                "TestProcess": {
                    "module": "test.testmodule",
                    "Parameters": {},
                    "save_file": None,
                    "plot_file": None
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
