import pytest
from numpy import allclose, array, arange, zeros

from skdh.preprocessing.wear_detection import (
    AccelThresholdWearDetection,
    DETACH,
    CountWearDetection,
)


class TestDETACH:
    def test(self, dummy_imu_data):
        time, accel, temperature, fs = dummy_imu_data

        detach = DETACH()

        res = detach.predict(time=time, accel=accel, temperature=temperature, fs=fs)

        # wear is only off by a little-bit from middle 3rd of data (120000 - 240000)
        assert allclose(res["wear"], array([[0, 119600], [240500, 360000]]))


class TestCountWearDetection:
    def test_nonwear_ends(self, np_rng):
        # generate some sample data
        fs = 30.0
        # make 7 hours of data
        t = arange(0, 3600 * 7, 1 / fs)
        x = zeros((t.size, 3))

        # 2hrs nonwear on each side, plus some time in the middle that will have
        # count = 0 with an interrupt but less than 90min total
        idx = (array([2, 3, 3.5, 4, 5, 5.4]) * 3600 * fs).astype(int)

        x[:, 2] = 1.0
        # noise
        x[idx[0] : idx[1], :] = np_rng.normal(scale=0.1, size=(idx[1] - idx[0], 3))
        x[idx[3] : idx[4], :] = np_rng.normal(scale=0.1, size=(idx[4] - idx[3], 3))

        # zero count interrupts
        n = int(60 * fs)
        x[idx[2] : idx[2] + n] = np_rng.normal(scale=0.05, size=(n, 3))
        x[idx[5] : idx[5] + n] = np_rng.normal(scale=0.05, size=(n, 3))

        # true wear array
        wear_true = array(
            [
                [
                    idx[0],
                    idx[-1] + n + n,
                ],  # need an extra minute of index because of windowing
            ]
        )

        cwd = CountWearDetection()

        res = cwd.predict(time=t, accel=x, fs=fs)

        assert allclose(res["wear"], wear_true)

    def test_nonwear_middle(self, np_rng):
        # generate some sample data
        fs = 30.0
        # make 7 hours of data
        t = arange(0, 3600 * 7, 1 / fs)
        x = zeros((t.size, 3))

        # 3hr block in middle of nonwear
        i1 = int(3600 * fs)
        i2 = int(3600 * 6 * fs)

        # set baseline accel values
        x[:, 2] = 1.0
        # add some noise
        x[:i1, :] = np_rng.normal(scale=0.1, size=(i1, 3))

        x[i2:, :] = np_rng.normal(scale=0.1, size=(t.size - i2, 3))

        # add some motion artefacts/<2min sensor knocks in the nonwear
        n = int(60 * fs)
        i = int(3600 * 3 * fs)
        # minute spike of data
        x[i : i + n, 0] = np_rng.normal(scale=0.15, size=n)

        # 34 min later (just outside 30min window)
        i = i + int(34 * 60 * fs)
        x[i : i + n, 1] = np_rng.normal(scale=0.15, size=n)

        # movement less than 30min before wear at end
        i3 = int(3600 * 5.6 * fs)
        x[i3 : i3 + n, 0] = np_rng.normal(scale=0.15, size=n)

        # create true wear array
        wear_true = array(
            [
                [
                    0,
                    i1 + int(60 * fs),
                ],  # have to add a second due to the way the windowing works
                [i3, t.size],
            ]
        )

        cwd = CountWearDetection()

        res = cwd.predict(time=t, accel=x, fs=fs)

        assert allclose(res["wear"], wear_true)


class TestDetectWearAccelThreshold:
    @pytest.mark.parametrize(("setup", "ship"), ((False, [0, 0]), (True, [12, 12])))
    def test(self, setup, ship, accel_with_nonwear):
        dw = AccelThresholdWearDetection(
            sd_crit=0.013,
            range_crit=0.05,
            apply_setup_criteria=setup,
            shipping_criteria=ship,
            window_length=60,
            window_skip=15,
        )

        time, accel, wear = accel_with_nonwear(setup, ship)

        res = dw.predict(time, accel)

        assert allclose(res["wear"], wear)

    @pytest.mark.parametrize(
        ("case", "setup", "ship"),
        (
            (1, False, [0, 0]),
            (1, True, [12, 12]),
            (2, False, [0, 0]),
            (2, True, [12, 12]),
            (3, False, [0, 0]),
            (3, True, [12, 12]),
            (4, False, [0, 0]),
            (4, True, [12, 12]),
        ),
    )
    def test_wear_time_modifiction(self, case, setup, ship, simple_nonwear_data):
        wskip = 15  # minutes

        nonwear, true_wear = simple_nonwear_data(case, wskip, setup, ship)

        starts, stops = AccelThresholdWearDetection._modify_wear_times(
            nonwear, wskip, setup, ship
        )

        assert allclose(starts, true_wear[:, 0])
        assert allclose(stops, true_wear[:, 1])

    def test_init(self):
        d = AccelThresholdWearDetection(shipping_criteria=[5, 10])
        assert d.ship_crit == [5, 10]

        d = AccelThresholdWearDetection(shipping_criteria=5)
        assert d.ship_crit == [5, 5]

        d = AccelThresholdWearDetection(shipping_criteria=True)
        assert d.ship_crit == [24, 24]
