import datetime as dt

import pytest
from numpy import array, allclose, zeros, arange
from numpy.random import default_rng

from skdh.activity.cutpoints import _base_cutpoints
from skdh.activity.core import ActivityLevelClassification
from skdh.activity import endpoints as epts
from skdh.gait import gait_metrics


class Test_update_date_results:
    @staticmethod
    def setup_results():
        k = [
            "Date",
            "Day Start Timestamp",
            "Day End Timestamp",
            "Weekday",
            "Day N",
            "N hours",
            "Total Minutes",
            "Wear Minutes",
        ]
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
            tzinfo=dt.timezone.utc,
        )
        epoch_ts = date.timestamp()

        time = [epoch_ts + i * 3600 for i in range(10)]

        ActivityLevelClassification()._update_date_results(res, time, 0, 0, 5, 0)

        assert res["Date"][0] == "2021-05-11"  # next day is when it starts
        assert res["Weekday"][0] == "Tuesday"
        assert res["Day N"][0] == 1
        assert res["N hours"][0] == 4.0
        assert res["Total Minutes"][0] == 4 * 60

    def test_late_day_start(self):
        res = self.setup_results()

        date = dt.datetime(
            year=2021,
            month=5,
            day=11,
            hour=10,
            minute=32,
            second=13,
            tzinfo=dt.timezone.utc,
        )
        epoch_ts = date.timestamp()

        time = [epoch_ts + i * 3600 for i in range(10)]

        ActivityLevelClassification()._update_date_results(res, time, 0, 0, 5, 12)

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

        c = {
            "light": 5,
            "sedentary": 2,
            "moderate": 10,
            "vigorous": 15,
            "metric": lambda x: x,
            "kwargs": {},
        }
        a = ActivityLevelClassification(cutpoints=c, day_window=None)
        assert a.cutpoints == c
        assert a.day_key == (-1, -1)

    def test_add(self):
        a = ActivityLevelClassification()
        # reset endpoints list
        a.wake_endpoints = []
        a.sleep_endpoints = []

        a.add(
            [
                epts.MaxAcceleration(5, state="wake"),
                epts.MaxAcceleration(5, state="sleep"),
            ]
        )
        a.add(epts.IntensityGradient(state="wake"))
        a.add(epts.IntensityGradient(state="sleep"))

        with pytest.warns(UserWarning):
            a.add([epts.MaxAcceleration(5, state="test")])
            a.add(epts.MaxAcceleration(5, state="test"))

        with pytest.warns(UserWarning):
            a.add(gait_metrics.GaitSpeedModel2())

        assert len(a.wake_endpoints) == 2
        assert len(a.sleep_endpoints) == 2

    def test(self, activity_res):
        a = ActivityLevelClassification(
            short_wlen=5,
            max_accel_lens=(10,),
            bout_lens=(10,),
            bout_criteria=0.8,
            bout_metric=4,
            min_wear_time=1,
            cutpoints="migueles_wrist_adult",
        )

        rng = default_rng(seed=5)
        x = zeros((240000, 3))
        x[:, 2] += rng.normal(loc=1, scale=1, size=x.shape[0])
        t = arange(1.6e9, 1.6e9 + x.shape[0] * 0.02, 0.02)

        sleep = array([[int(0.8 * t.size), t.size - 1]])

        res = a.predict(time=t, accel=x, fs=None, wear=None, sleep=sleep)

        for k in activity_res:
            assert allclose(res[k], activity_res[k], equal_nan=True)
