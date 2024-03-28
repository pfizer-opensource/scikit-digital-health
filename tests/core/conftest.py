import pytest

from skdh.base import BaseProcess, handle_process_returns
from skdh import __version__ as skdh_vers


@pytest.fixture(scope="module")
def testprocess():
    class TestProcess(BaseProcess):
        def __init__(self, kw1=1):
            super().__init__(kw1=kw1)
            self.kw1 = kw1

        def predict(self, *args, **kwargs):
            self.logger.info(f"kw1={self.kw1}")

            return {"kw1": self.kw1}, {"kw1": self.kw1}

    return TestProcess


@pytest.fixture(scope="module")
def testprocess2():
    class TestProcess2(BaseProcess):
        def __init__(self, kwa=5):
            super().__init__(kwa=kwa)
            self.kwa = kwa

        def predict(self, *args, **kwargs):
            self.logger.info(f"kwa={self.kwa}")

            return {"kwa": self.kwa}, {"kwa": self.kwa}

    return TestProcess2


@pytest.fixture(scope="module")
def testprocess3():
    class TestProcess3(BaseProcess):
        def __init__(self, kwa=1):
            super().__init__(kwa=kwa)
            self.kwa = kwa

        @handle_process_returns(results_to_kwargs=True)
        def predict(self, a=None, b=None, *, c=None, **kwargs):
            super().predict(
                expect_days=False, expect_wear=False, a=a, b=b, c=c, **kwargs
            )

            return {"d": 5.5, "e": 8}

    return TestProcess3


@pytest.fixture(scope="module")
def dummy_pipeline():
    exp = {
        "Steps": [
            {
                "name": "Gait",
                "process": "GaitLumbar",
                "package": "skdh",
                "module": "gait.core",
                "parameters": {},
                "save_file": "gait_results.csv",
                "plot_file": None,
            },
            {
                "name": "TestProcess",
                "process": "TestProcess",
                "package": "skdh",
                "module": "test.testmodule",
                "parameters": {},
                "save_file": None,
                "plot_file": None,
            },
        ],
        "Version": skdh_vers,
    }

    return exp
