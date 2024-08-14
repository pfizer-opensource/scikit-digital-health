import pytest
from numpy import arange, diff, allclose

from skdh.preprocessing import FillGaps


class TestFillGaps:
    def test(self, np_rng):
        # make some dummy data
        t = arange(0, 50, 1)
        t[10:] += 10  # simulate a 10-second gap in the data
        t[38:] += 15  # simulate a 15-second gap in the data

        fs = 1.0
        acc = np_rng.random((t.size, 3))
        temp = np_rng.random(t.size) + 27

        # single stream data
        t1 = arange(0, 45, 0.5)
        t1[20:] += 12
        ds_ = np_rng.random(t1.size) * 0.01
        ds = {"time": t1, "fs": 2.0, "values": ds_}

        # second one which WON'T be filled
        ds2 = {"time": t1, "fs": 2.0, "values": ds_ * 2.0}

        fg = FillGaps(
            fill_values={
                "accel": [0.0, 0.0, 1.0],
                "ds": -500.0,
                # purposely include temperature which has a default fill value
                # DON'T include 'ds2' which won't be filled
            }
        )

        res = fg.predict(time=t, fs=fs, accel=acc, temperature=temp, ds=ds, ds2=ds2)

        # check the results
        assert res["time"][-1] == (49 + 10 + 15)
        assert res["time"].size == 50 + 10 + 15
        assert diff(res["time"]).max() == 1.0
        assert res["accel"].shape == (50 + 10 + 15, 3)
        assert res["temperature"].shape == (50 + 10 + 15,)

        assert res["ds"]["time"].size == (45 + 12) * 2  # 2x because of the 0.5 fs
        assert diff(res["ds"]["time"]).max() == 0.5
        assert res["ds"]["values"].shape == ((45 + 12) * 2,)
        assert res["ds"]["values"].min() == -500.0

        assert "ds2" not in res

    def test_nogap(self, np_rng):
        # make some dummy data
        t = arange(0, 50, 1)
        fs = 1.0
        acc = np_rng.random((t.size, 3))
        temp = np_rng.random(t.size) + 27

        fg = FillGaps(
            fill_values={
                "accel": [0.0, 0.0, 1.0],
            }
        )

        res = fg.predict(
            time=t,
            fs=fs,
            accel=acc,
            temperature=temp,
        )

        # check the results
        assert res["time"][-1] == 49
        assert res["time"].size == 50
        assert diff(res["time"]).max() == 1.0
        assert res["accel"].shape == (50, 3)
        assert res["temperature"].shape == (50,)

        assert allclose(res["accel"], acc)
