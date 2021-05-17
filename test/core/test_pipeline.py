from skimu.pipeline import Pipeline, NotAProcessError, ProcessNotFoundError
from skimu.base import _BaseProcess


class TestPipeline:
    def test_str_repr(self):
        # temp class just for testing
        class TestProcess(_BaseProcess):
            def __init__(self, kw1=1):
                super().__init__(kw1=kw1)

        p = Pipeline()

        assert repr(p) == "IMUAnalysisPipeline[\n]"

        p.add(TestProcess(kw1=2))
        p.add(TestProcess(kw1=1))

        assert str(p) == "IMUAnalysisPipeline"
        assert repr(p) == "IMUAnalysisPipeline[\n\tTestProcess(kw1=2),\n\tTestProcess(kw1=1),\n]"
