import datetime as dt

import pytest
from numpy import array, allclose

from skdh.activity.cutpoints import _base_cutpoints
from skdh.activity.core import (
    _update_date_results,
    ActivityLevelClassification,
    get_activity_bouts,
    get_intensity_gradient,
)


class Test_update_date_results:
    @staticmethod
    def setup_results():
        k = ["Date", "Weekday", "Day N", "N hours"]
        r = {i: [0] for i in k}

        return r

    def test(self):
        res = self.setup_results()

        date = dt.datetime(
            year=2021,
            month=5,
            day=10,
            hour=23,
            minute=59,
            second=57,
            tzinfo=dt.timezone.utc
        )
        epoch_ts = date.timestamp()

        time = [epoch_ts + i * 3600 for i in range(10)]

        _update_date_results(res, time, 0, 0, 5, 0)

        assert res["Date"][0] == "2021-05-11"  # next day is when it starts
        assert res["Weekday"][0] == "Tuesday"
        assert res["Day N"][0] == 1
        assert res["N hours"][0] == 4.0

    def test_late_day_start(self):
        res = self.setup_results()

        date = dt.datetime(
            year=2021,
            month=5,
            day=11,
            hour=10,
            minute=32,
            second=13,
            tzinfo=dt.timezone.utc
        )
        epoch_ts = date.timestamp()

        time = [epoch_ts + i * 3600 for i in range(10)]

        _update_date_results(res, time, 0, 0, 5, 12)

        # previous day is the actual window start
        assert res["Date"][0] == "2021-05-10"
        assert res["Weekday"][0] == "Monday"
        assert res["Day N"][0] == 1
        assert res["N hours"][0] == 4.0


class TestActivityLevelClassification:
    def test_init(self):
        # make sure wlen gets sent to factor of 60
        swlen_opts = [(4, 4), (7, 6), (14, 15), (58, 30)]
        for wlen in swlen_opts:
            if wlen[0] != wlen[1]:
                with pytest.warns(UserWarning):
                    a = ActivityLevelClassification(short_wlen=wlen[0])
            else:
                a = ActivityLevelClassification(short_wlen=wlen[0])

            assert a.wlen == wlen[1]

        # check that default cutpoints are set if not provided
        with pytest.warns(UserWarning):
            a = ActivityLevelClassification(cutpoints="test")

        # dont have name so make sure that the values are correct
        assert a.cutpoints == _base_cutpoints["migueles_wrist_adult"]
