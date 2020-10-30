import pytest

from skimu.base import _BaseProcess


class TestBaseProcess:
    def test_init_repr_str(self):
        bp = _BaseProcess('Process', True, kw1=None, kw2=0, kw3=0.5, kw4='str', kw5=False)

        r = "Process(kw1=None, kw2=0, kw3=0.5, kw4='str', kw5=False)"

        assert bp._proc_name == "Process"
        assert bp._return_result
        assert repr(bp) == r
        assert str(bp) == "Process"
