import pytest
from numpy import isclose, allclose, nan, array, concatenate, log
from numpy.random import default_rng

from skdh.activity.endpoints import (
    get_activity_bouts,
    ActivityEndpoint,
    IntensityGradient,
    MaxAcceleration,
    TotalIntensityTime,
    BoutIntensityTime,
    FragmentationEndpoints,
)


class Test_get_activity_bouts:
    def test(self, act_acc):
        res_1_nclosed = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 1)
        res_1_closed = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, False, 1)
        res_2 = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 2)
        res_3 = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 3)
        res_4 = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 4)

        true_1_nclosed = 21.0
        true_1_closed = 20.0
        true_2 = 24.0
        true_3 = 22.0
        true_4 = 22.0

        assert isclose(res_1_nclosed, true_1_nclosed)
        assert isclose(res_1_closed, true_1_closed)
        assert isclose(res_2, true_2)
        assert isclose(res_3, true_3)
        assert isclose(res_4, true_4)

    def test_oob_boutmetric(self, act_acc):
        with pytest.raises(ValueError):
            get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, False, 5)

    def test_moving_mean_valueerror(self):
        x = array([1, 1, 1, 0])

        r = get_activity_bouts(x, 0.5, 1.5, 60, 6, 0.8, False, 3)
        assert isclose(r, 0.0)

        r = get_activity_bouts(x, 0.5, 1.5, 60, 6, 0.8, False, 4)
        assert isclose(r, 0.0)


class TestActivityEndpoint:
    def test_init(self):
        a = ActivityEndpoint("test", "wake")

        assert a.name == "wake test"

        b = ActivityEndpoint(["t1", "t2"], "wake")

        assert b.name == ["wake t1", "wake t2"]

    def test_predict(self):
        a = ActivityEndpoint("test", "wake")
        a.predict()

    def test_reset_cached(self):
        a = ActivityEndpoint("test", "wake")
        a.reset_cached()


class TestIntensityGradient:
    def test(self, act_results):
        ig = IntensityGradient()
        # generate 10,000 random samples from uniform distribution
        a = default_rng(seed=5).uniform(0, 4.0, 20000)
        # add values between 4-8 so we get a flat line
        a = concatenate((a, [6.0] * 125))
        b = a[::12]

        ig.predict(act_results, 0, a, b, 5, 12)

        assert act_results["wake intensity gradient"] == [0.0]
        ig.reset_cached()

        # account for variation with the uniform sample
        assert allclose(act_results["wake intensity gradient"], 0.0, atol=0.01)
        assert allclose(act_results["wake ig intercept"], log(10.4), atol=0.04)

    def test_nan(self, act_results):
        ig = IntensityGradient()
        ig.predict(
            act_results, 0, array([0.2, 0.2, 0.2, 0.2]), array([0.2, 0.2]), 5, 12
        )

        ig.reset_cached()

        assert allclose(act_results["wake intensity gradient"], nan, equal_nan=True)


class TestMaxAcceleration:
    def test_init(self):
        a = MaxAcceleration(5)
        assert a.wlens == [5]
        assert a.name == ["wake max acc 5min [g]"]

    def test(self, act_results):
        a = default_rng(seed=5).normal(size=500)
        b = default_rng(seed=10).normal(loc=30, size=500)

        ma = MaxAcceleration(2)
        # check that it doesnt change if too little accel provided
        ma.predict(
            act_results, 0, array([0.2, 0.2, 0.2, 0.2]), array([0.2, 0.2]), 5, 12
        )
        assert allclose(act_results["wake max acc 2min [g]"], 0)

        ma.predict(act_results, 0, a, a[::12], 5, 12)
        assert allclose(act_results["wake max acc 2min [g]"], 0, atol=1)

        ma.predict(act_results, 0, b, b[::12], 5, 12)
        assert allclose(act_results["wake max acc 2min [g]"], 30, atol=1)


class TestTotalIntensityTime:
    def test_init(self):
        with pytest.warns(UserWarning):
            a = TotalIntensityTime("MVPA", 5, cutpoints=None)
        assert isclose(a.lthresh, 0.110)

        a = TotalIntensityTime("MVPA", 5, cutpoints="vaha-ypya_hip_adult")
        assert isclose(a.lthresh, 0.091)

        with pytest.raises(ValueError):
            TotalIntensityTime("MVPA", 5, cutpoints={"light": 0.5})

        with pytest.raises(ValueError):
            TotalIntensityTime("MVPA", 5, cutpoints=5.0)

    def test(self, act_results):
        a = array([0.0, 0.2, 0.2, 0.2, 0.5, 0.5, 1.5, 0])
        b = array([0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 1.5])

        e = TotalIntensityTime("MVPA", 5)
        e.predict(act_results, 0, a, a[::4], 5, 12)
        assert allclose(act_results["wake MVPA 5s epoch [min]"], 0.5)

        e.predict(act_results, 0, b, b[::4], 5, 12)
        assert allclose(act_results["wake MVPA 5s epoch [min]"], 0.75)


class TestBoutIntensityTime:
    def test_init(self):
        with pytest.warns(UserWarning):
            a = BoutIntensityTime("MVPA", 6, 0.8, 4, False, cutpoints=None)
        assert isclose(a.lthresh, 0.110)

        a = BoutIntensityTime("MVPA", 6, 0.8, 4, False, cutpoints="vaha-ypya_hip_adult")
        assert isclose(a.lthresh, 0.091)

        with pytest.raises(ValueError):
            BoutIntensityTime("MVPA", 6, 0.8, 4, False, cutpoints={"light": 0.5})

        with pytest.raises(ValueError):
            BoutIntensityTime("MVPA", 6, 0.8, 4, False, cutpoints=5.0)

    def test(self, act_results, act_acc):
        c = {
            "metric": lambda x: x,
            "kwargs": {},
            "sedentary": 0.25,
            "light": 0.5,
            "moderate": 1.0,
        }
        e = BoutIntensityTime("MVPA", 6, 0.8, 4, False, c)

        e.predict(act_results, 0, act_acc, act_acc[::1], 60, 1)

        assert allclose(act_results["wake MVPA 6min bout [min]"], 22.0)


class TestFragmentationEndpoints:
    def test_init(self):
        with pytest.warns(UserWarning):
            a = FragmentationEndpoints("MVPA", cutpoints=None)
        assert isclose(a.lthresh, 0.110)

        a = FragmentationEndpoints("MVPA", cutpoints="vaha-ypya_hip_adult")
        assert isclose(a.lthresh, 0.091)

        with pytest.raises(ValueError):
            a = FragmentationEndpoints("MVPA", cutpoints={"light": 0.5})

        with pytest.raises(ValueError):
            FragmentationEndpoints("MVPA", cutpoints=5.0)

    def test(self, act_results, dummy_frag_predictions):
        e = FragmentationEndpoints("MVPA", cutpoints="migueles_wrist_adult")

        e.predict(act_results, 0, dummy_frag_predictions, dummy_frag_predictions, 5, 12)
        e.reset_cached()

        assert allclose(act_results["wake MVPA avg duration"], 13 / 3)
        assert allclose(act_results["wake MVPA transition probability"], 3 / 13)
        assert allclose(act_results["wake MVPA gini index"], 0.3076923)
        assert allclose(act_results["wake MVPA avg hazard"], 0.8333333)
        assert allclose(act_results["wake MVPA power law distribution"], 3.151675)
