import json
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

from skimu.pipeline import Pipeline, NotAProcessError, ProcessNotFoundError
from skimu.gait import Gait


class TestPipeline:
    @staticmethod
    def setup_lgr():
        class Lgr:
            msgs = []

            def info(self, msg):
                self.msgs.append(msg)

        return Lgr()

    def test_run(self, testprocess, testprocess2):
        p = Pipeline()

        tp1 = testprocess(kw1=1)
        tp1.logger = self.setup_lgr()
        tp2 = testprocess2(kwa=5)
        tp2.logger = self.setup_lgr()

        p.add(tp1)
        p.add(tp2)

        res = p.run()

        assert "kw1=1" in tp1.logger.msgs
        assert "kwa=5" in tp2.logger.msgs

        exp_res = {"TestProcess": {"kw1": 1}, "TestProcess2": {"kwa": 5}}

        assert res == exp_res

    def test_str_repr(self, testprocess):
        p = Pipeline()

        assert repr(p) == "IMUAnalysisPipeline[\n]"

        p.add(testprocess(kw1=2))
        p.add(testprocess(kw1=1))

        assert str(p) == "IMUAnalysisPipeline"
        assert repr(p) == "IMUAnalysisPipeline[\n\tTestProcess(kw1=2),\n\tTestProcess(kw1=1),\n]"

    def test_add(self, testprocess):
        p = Pipeline()

        tp = testprocess(kw1=500)
        p.add(tp, save_file="test_saver.csv")

        assert p._steps == [testprocess(kw1=500)]
        assert tp._in_pipeline
        assert tp.pipe_save_file == "test_saver.csv"

        with pytest.raises(NotAProcessError):
            p.add(list())

    def test_save(self, testprocess):
        p = Pipeline()

        tp = testprocess(kw1=2)
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
