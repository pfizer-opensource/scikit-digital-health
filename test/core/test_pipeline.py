from tempfile import TemporaryDirectory
from pathlib import Path

# DEPRECATED
import json

import pytest
import yaml

from skdh.pipeline import Pipeline, NotAProcessError, ProcessNotFoundError, VersionError
from skdh.gait import Gait
from skdh import __version__ as skdh_vers


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
        assert (
            repr(p)
            == "IMUAnalysisPipeline[\n\tTestProcess(kw1=2),\n\tTestProcess(kw1=1),\n]"
        )

    def test_add(self, testprocess):
        p = Pipeline()
        p2 = Pipeline()

        tp = testprocess(kw1=500)
        tp2 = testprocess(kw1=400)
        p.add(tp, save_file="test_saver.csv", make_copy=False)
        p2.add(tp2, save_file="test_saver.csv")

        assert p._steps[0] is tp
        assert p._steps == [testprocess(kw1=500)]
        assert tp._in_pipeline
        assert tp.pipe_save_file == "test_saver.csv"

        # check when adding a copy
        assert tp2 is not p2._steps[0]
        assert not tp2._in_pipeline  # should be set on the copy
        assert tp2.pipe_save_file is None
        assert p2._steps[0]._in_pipeline

        with pytest.raises(NotAProcessError):
            p.add(list())

    def test_save(self, testprocess):
        p = Pipeline()

        tp = testprocess(kw1=2)
        # overwrite this for saving
        tp.__class__.__module__ = "skdh.test.testmodule"

        p.add(tp)

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.skdh"

            p.save(str(fname))
            with fname.open() as f:
                res = yaml.safe_load(f)

        exp = {
            "Steps": [
                {
                    "TestProcess": {
                        "package": "skdh",
                        "module": "test.testmodule",
                        "parameters": {"kw1": 2},
                        "save_file": None,
                        "plot_file": None,
                    }
                }
            ],
            "Version": skdh_vers,
        }

        assert res == exp

    def test__handle_load_input(self, dummy_pipeline):
        with TemporaryDirectory() as tdir:
            fname1 = Path(tdir) / "file.random"
            fname_json = Path(tdir) / "file.json"
            fname_yaml = Path(tdir) / "file.yaml"

            with fname1.open(mode="w") as f:
                yaml.dump(dummy_pipeline, f)
            with fname_json.open(mode="w") as f:
                json.dump(dummy_pipeline, f)
            with fname_yaml.open(mode="w") as f:
                f.write("unbalanced brackets: ][\n")

            with pytest.warns(
                UserWarning, match="does not have one of the expected suffixes:"
            ):
                Pipeline._handle_load_input(None, None, str(fname1))
            with pytest.warns(
                DeprecationWarning, match="JSON format will be deprecated"
            ):
                Pipeline._handle_load_input(None, None, file=str(fname_json))
            # handle the json reader failing as well
            with pytest.raises(json.decoder.JSONDecodeError):
                with pytest.warns(UserWarning, match="Error reading file as YAML"):
                    Pipeline._handle_load_input(None, None, file=str(fname_yaml))

    def test_load_through_init(self, dummy_pipeline):
        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.skdh"

            with fname.open(mode="w") as f:
                yaml.dump(dummy_pipeline, f)

            with pytest.warns(UserWarning):
                p = Pipeline(dict(file=str(fname)))

        assert p._steps == [Gait()]

        # Test loading from a yaml string
        dummy_pipe_str = yaml.dump(dummy_pipeline)
        p2 = Pipeline(dict(yaml_str=dummy_pipe_str))

        assert p2._steps == [Gait()]

    def test_load_function(self, dummy_pipeline):
        p = Pipeline()
        p2 = Pipeline()

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.skdh"

            with fname.open(mode="w") as f:
                # save only the steps to trigger version warning
                yaml.dump(dummy_pipeline["Steps"], f)

            with pytest.warns(
                UserWarning, match="Pipeline created by an unknown version"
            ):
                p.load(str(fname))

        assert p._steps == [Gait()]

        pipe_str = yaml.dump(dummy_pipeline)
        p2.load(yaml_str=pipe_str)

        assert p2._steps == [Gait()]

        # check for the deprecation warning for json input
        pipe_str_json = json.dumps(dummy_pipeline)
        with pytest.warns(DeprecationWarning):
            p2.load(json_str=pipe_str_json)

    def test_load_errors(self, dummy_pipeline):
        p = Pipeline()

        pipe_str = yaml.dump(dummy_pipeline)
        pipe_str_steps_only = yaml.dump(dummy_pipeline["Steps"])

        with pytest.raises(VersionError):
            p.load(yaml_str=pipe_str_steps_only, noversion_raise=True)

        with pytest.raises(VersionError):
            p._min_vers = "100.0.0"
            p.load(yaml_str=pipe_str, old_raise=True)

        with pytest.raises(ProcessNotFoundError):
            p.load(yaml_str=pipe_str, process_raise=True)

    def test_load_version_warning(self, dummy_pipeline):
        p = Pipeline()
        p._min_vers = "100.0.0"

        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "file.yaml"

            with fname.open(mode="w") as f:
                yaml.dump(dummy_pipeline, f)

            with pytest.warns(
                UserWarning, match="Pipeline was created by an older version of skdh"
            ):
                p.load(str(fname))
