from numpy import array, allclose
from skimu.base import _BaseProcess


class Test_BaseProcess:
    @staticmethod
    def setup_lgr():
        class Lgr:
            msgs = []

            def info(self, msg):
                self.msgs.append(msg)

        return Lgr()

    def test__check_if_idx_none(self):
        lgr = self.setup_lgr()
        bp = _BaseProcess()
        bp.logger = lgr  # overwrite the logger

        x = array([[0, 10], [15, 20]])

        s, e = bp._check_if_idx_none(x, "none msg", None, None)

        assert allclose(s, [0, 15])
        assert allclose(e, [10, 20])

        s, e = bp._check_if_idx_none(None, "none msg", 0, 10)
        sn, en = bp._check_if_idx_none(None, "none msg", None, 10)

        assert "none msg" in lgr.msgs
        assert s == 0
        assert e == 10

        assert sn is None
        assert en is None

    def test_str_repr(self):
        bp = _BaseProcess(kw1=1, kw2="2")

        assert str(bp) == "_BaseProcess"
        assert repr(bp) == "_BaseProcess(kw1=1, kw2='2')"
