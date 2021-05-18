import pytest

from skimu.base import _BaseProcess


@pytest.fixture(scope="module")
def testprocess():
    class TestProcess(_BaseProcess):
        def __init__(self, kw1=1):
            super().__init__(kw1=kw1)
            self.kw1 = kw1

        def predict(self, *args, **kwargs):
            self.logger.info(f"kw1={self.kw1}")

            return {"kw1": self.kw1}, {"kw1": self.kw1}

    return TestProcess


@pytest.fixture(scope="module")
def testprocess2():
    class TestProcess2(_BaseProcess):
        def __init__(self, kwa=5):
            super().__init__(kwa=kwa)
            self.kwa = kwa

        def predict(self, *args, **kwargs):
            self.logger.info(f"kwa={self.kwa}")

            return {"kwa": self.kwa}, {"kwa": self.kwa}

    return TestProcess2
