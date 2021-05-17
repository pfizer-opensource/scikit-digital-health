from numpy import array, allclose
from skimu.base import _BaseProcess


class Test_BaseProcess:
    def test_str_repr(self):
        bp = _BaseProcess(kw1=1, kw2="2")

        assert str(bp) == "_BaseProcess"
        assert repr(bp) == "_BaseProcess(kw1=1, kw2='2')"

    @staticmethod
    def setup_lgr():
        class Lgr:
            msgs = []

            def info(self, msg):
                self.msgs.append(msg)

        return Lgr()

    def test__check_if_idx_none(self):
        bp = _BaseProcess()
        bp.logger = self.setup_lgr()  # overwrite the logger

        x = array([[0, 10], [15, 20]])

        s, e = bp._check_if_idx_none(x, "none msg", None, None)

        assert allclose(s, [0, 15])
        assert allclose(e, [10, 20])

        s, e = bp._check_if_idx_none(None, "none msg", 0, 10)
        sn, en = bp._check_if_idx_none(None, "none msg", None, 10)

        assert "none msg" in bp.logger.msgs
        assert s == 0
        assert e == 10

        assert sn is None
        assert en is None

    def test_predict(self):
        bp = _BaseProcess()
        bp.logger = self.setup_lgr()

        bp.predict(
            expect_days=True,
            expect_wear=True,
            accel=array([[1, 2, 3], [4, 5, 6]])
        )

        assert bp._file_name == ""
        assert "Entering _BaseProcess processing with call _BaseProcess()" in bp.logger.msgs
        assert "[_BaseProcess] Day indices [(-1, -1)] not found. No day split used." in bp.logger.msgs

