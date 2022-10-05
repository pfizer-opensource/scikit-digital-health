import pytest
from numpy import allclose, array

from skdh.preprocessing.wear_detection import AccelThresholdWearDetection, DETACH, CountWearDetection


class TestDETACH:
    def test(self, dummy_imu_data):
        time, accel, temperature, fs = dummy_imu_data

        detach = DETACH()

        res = detach.predict(time=time, accel=accel, temperature=temperature, fs=fs)

        # wear is only off by a little-bit from middle 3rd of data (120000 - 240000)
        assert allclose(res['wear'], array([[0, 119600], [240500, 360000]]))


class TestCountWearDetection:
    def test(self, dummy_imu_data):
        time, accel, _, fs = dummy_imu_data

        cwd = CountWearDetection()

        res = cwd.predict(time=time, accel=accel, fs=fs)

        assert True


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
