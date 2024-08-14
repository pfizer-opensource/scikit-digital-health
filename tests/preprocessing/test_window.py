from datetime import datetime, timezone

import pytest
from numpy import ndarray, int_, arange, allclose
import pandas as pd

from skdh.preprocessing import GetDayWindowIndices


class TestGetDayWindowIndices:
    def test(self):
        t1 = datetime(
            year=2020,
            month=7,
            day=5,
            hour=14,
            minute=38,
            second=19,
            tzinfo=timezone.utc,
        )
        t2 = datetime(
            year=2020,
            month=7,
            day=10,
            hour=11,
            minute=1,
            second=59,
            tzinfo=timezone.utc,
        )

        t = arange(t1.timestamp(), t2.timestamp(), 1.0, dtype=int_)

        days = GetDayWindowIndices(bases=[0, 15, 8], periods=[24, 4, 2]).predict(
            time=t, fs=1.0
        )

        assert allclose(
            days["day_ends"][(0, 24)],
            [
                [0, 33701],
                [33701, 120101],
                [120101, 206501],
                [206501, 292901],
                [292901, 379301],
                [379301, 419019],
            ],
        )

        assert allclose(
            days["day_ends"][(15, 4)],
            [
                [1301, 15701],
                [87701, 102101],
                [174101, 188501],
                [260501, 274901],
                [346901, 361301],
            ],
        )

        assert allclose(
            days["day_ends"][(8, 2)],
            [
                [62501, 69701],
                [148901, 156101],
                [235301, 242501],
                [321701, 328901],
                [408101, 415301],
            ],
        )

    def test_dst(self):
        dr = pd.date_range(
            start="2023-11-04 18:00:00",
            end="2023-11-06 02:00:00",
            freq="s",
            tz="US/Eastern",
        )
        time = dr.astype(int).values / 1e9

        days = GetDayWindowIndices(bases=[0], periods=[24]).predict(
            time=time, fs=1.0, tz_name="US/Eastern"
        )

        assert allclose(
            days["day_ends"][(0, 24)],
            [[0, 21600], [21600, 111600], [111600, time.size - 1]],
        )

    def test_window_inputs(self):
        w = GetDayWindowIndices(bases=None, periods=None)
        assert not w.window
        assert isinstance(w.bases, ndarray)
        assert isinstance(w.periods, ndarray)

        with pytest.warns(UserWarning) as record:
            w = GetDayWindowIndices(bases=8, periods=None)
            w = GetDayWindowIndices(bases=None, periods=12)

        assert len(record) == 2
        assert "One of base or period is None" in record[0].message.args[0]
        assert "One of base or period is None" in record[1].message.args[0]

    def test_window_range_error(self):
        with pytest.raises(ValueError):
            GetDayWindowIndices(bases=[0, 24], periods=[5, 26])
