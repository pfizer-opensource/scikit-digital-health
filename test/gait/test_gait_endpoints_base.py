import pytest
from numpy import array, allclose, arange, nan, isnan

from skimu.gait import gait_endpoints as endpoints


class TestEventEndpoint1(endpoints.GaitEventEndpoint):
    def __init__(self):
        super().__init__("test event ept 1", __name__, depends=None)

    def _predict(self, *args, **kwargs):
        print("inside event endpoint 1")


class TestEventEndpoint2(endpoints.GaitEventEndpoint):
    def __init__(self):
        super().__init__("test event ept 2", __name__, depends=[TestEventEndpoint1])

    def _predict(self, *args, **kwargs):
        print("inside event endpoint 2")


class TestBoutEndpoint1(endpoints.GaitBoutEndpoint):
    def __init__(self):
        super().__init__("test bout ept 1", __name__, depends=[TestEventEndpoint2])

    def _predict(self, *args, **kwargs):
        print("inside bout endpoint 1")


def test_depends(capsys):
    tbe1 = TestBoutEndpoint1()

    tbe1.predict(50., 1.8, {}, {})

    record = capsys.readouterr().out

    exp = "inside event endpoint 1\ninside event endpoint 2\ninside bout endpoint 1"
    assert exp in record


class TestBoutEndpoint:
    def test_already_run(self, capsys):
        tbe1 = TestBoutEndpoint1()
        tbe1.predict(50., 1.8, {tbe1.k_: []}, {})

        record = capsys.readouterr().out

        assert "inside bout endpoint 1" not in record


class TestEventEndpoint:
    def test_already_run(self, capsys):
        tee1 = TestEventEndpoint1()
        tee1.predict(50., 1.8, {tee1.k_: []}, {})

        record = capsys.readouterr().out

        assert "inside event endpoint 1" not in record

    def test__get_mask(self):
        gait = {
            "IC": array([10, 20, 30, 40, 50, 60, 70, 80]),
            "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2])
        }

        mask = endpoints.GaitEventEndpoint._get_mask(gait, 1)

        assert allclose(mask, [True, True, False, True, True, True, True, False])

        with pytest.raises(ValueError):
            endpoints.GaitEventEndpoint._get_mask(gait, 5)

    def test__predict_asymmetry(self):
        gait = {
            "IC": array([10, 20, 30, 40, 50, 60, 70, 80]),
            "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2]),
            "PARAM:test event ept 1": arange(1, 9)
        }

        tee1 = TestEventEndpoint1()
        tee1._predict_asymmetry(50., 1.8, gait, {})

        res = gait['PARAM:test event ept 1 asymmetry']
        exp = array([1, 1, nan, 1, 1, 1, 1, nan], dtype="float")

        assert allclose(res, exp, equal_nan=True)

    def test__predict_init(self):
        gait = {
            "IC": array([10, 20, 30, 40, 50, 60, 70, 80]),
            "Bout N": array([1, 1, 1, 2, 2, 2, 2, 2])
        }

        tee1 = TestEventEndpoint1()

        m, mo = tee1._predict_init(gait, init=True, offset=1)

        assert "PARAM:test event ept 1" in gait
        assert gait["PARAM:test event ept 1"].size == 8
        assert all(isnan(gait["PARAM:test event ept 1"]))
