import pytest
from numpy import array, allclose, arange, pi, sin, cos, zeros, isclose
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt

from skdh.sit2stand.detector import pad_moving_sd, get_stillness, Detector


def test_pad_moving_sd():
    x = arange(0, 10)

    mn, sd, pad = pad_moving_sd(x, 3, 1)

    assert mn.size == sd.size == 10
    assert pad == 2
    assert allclose(mn, [1, 1, 1, 2, 3, 4, 5, 6, 7, 8])
    assert allclose(sd, 1.0)


def test_get_stillness(dummy_stillness_data):
    threshs = {
        "accel moving avg": 0.2,
        "accel moving std": 0.1,
        "jerk moving avg": 2.5,
        "jerk moving std": 3.0,
    }
    still, starts, stops, long_starts, long_stops = get_stillness(
        dummy_stillness_data * 9.81,
        1 / 20,
        9.81,
        0.25,
        0.5,
        threshs,
    )

    assert 300 <= starts[0] <= 304  # account for padding
    assert 397 <= stops[0] <= 400  # account for padding
    assert 94 <= still.sum() <= 100  # padding

    assert long_starts == starts  # 100 samples -> 5 seconds
    assert long_stops == stops


class TestDetector:
    def test(self, np_rng):
        t_ = arange(0, 10, 0.01)
        acc = zeros((5000, 3))
        acc[:, 2] += 1
        acc[2000:2500, 2] += (0.33 * t_ ** (1 / 3) * sin(2 * pi * 0.1 * t_))[::2]
        acc += np_rng.standard_normal(acc.shape) * 0.03

        sos = butter(4, 2 * 1.5 / 100, btype="low", output="sos")
        acc = sosfiltfilt(sos, acc, axis=0)

        res = {
            "Date": [],
            "Day Number": [],
            "Time": [],
            "Hour": [],
            "STS Start": [],
            "STS End": [],
            "Duration": [],
            "Max. Accel.": [],
            "Min. Accel.": [],
            "SPARC": [],
            "Vertical Displacement": [],
            "Partial": [],
        }
        acc_f = norm(acc, axis=1)

        Detector().predict(
            res, 0.01, arange(0, 20, 0.004), acc, acc_f, array([2200]), tz_name="UTC"
        )

        # 2 seconds, off due to detection of still periods, etc
        assert isclose(res["Duration"][0], 1.92, atol=5e-2)
        # large ranges due to random noise
        assert isclose(res["Max. Accel."][0], 14.2, atol=0.3)
        assert isclose(res["Min. Accel."][0], 3.4, atol=0.2)
        assert isclose(res["SPARC"][0], -1.9, atol=0.2)
        # this is off so much because of synthetic data
        assert isclose(res["Vertical Displacement"][0], 20.4, atol=0.5)
        assert not res["Partial"][0]

    def test_update_thresh(self):
        d = Detector(thresholds={"accel moving avg": 0.5, "test": 100})

        assert d.thresh["accel moving avg"] == 0.5

        for k in d._default_thresholds:
            if k != "accel moving avg":
                assert d.thresh[k] == d._default_thresholds[k]

    def test__get_vertical_accel(self, np_rng):
        d = Detector(gravity_pass_order=4, gravity_pass_cutoff=0.5)

        x = zeros((500, 3)) + np_rng.standard_normal((500, 3)) * 0.02
        x[:, 2] += 1

        vacc = d._get_vertical_accel(0.05, x)

        assert allclose(vacc, x[:, 2], atol=5e-3)

    def test__integrate(self):
        dt = 0.01
        c = 2 * pi * 0.2
        t = arange(0, 5.001, dt)  # 501 samples
        a = sin(c * t)

        vt = (-cos(c * t) + 1) / c
        pt = (c * t - sin(c * t)) / c**2

        v, p = Detector._integrate(a, dt, True)
        v1, p1 = Detector._integrate(a, dt, False)

        assert allclose(v, vt, atol=1e-5)
        assert allclose(p, pt, atol=5e-5)

        assert allclose(v1, v)
        assert allclose(p1, p)

    def test__get_end_still(self):
        time = arange(0, 10, 0.01)
        # STILLNESS
        d = Detector(stillness_constraint=True)

        e_still = d._get_end_still(time, array([125, 225]), array([125]), 150)

        assert e_still == 125

        with pytest.raises(IndexError):
            d._get_end_still(time, array([125]), array([125]), time.size - 50)

        # NO STILLNESS
        d = Detector(stillness_constraint=False)

        e_still = d._get_end_still(time, array([125, 225]), array([125]), 150)

        assert e_still == 125

        time = arange(0, 40, 0.1)

        with pytest.raises(IndexError):
            d._get_end_still(time, array([10]), array([10]), time.size - 10)

    def test__get_start_still(self):
        time = arange(0, 10, 0.01)

        d = Detector()

        es, sae = d._get_start_still(0.01, time, array([125, 350]), array([125]), 100)

        assert es == 125
        assert sae

        time = arange(0, 40, 0.1)
        peak = 50
        es, sae = d._get_start_still(0.1, time, array([370]), array([370]), peak)

        assert es == peak + 50  # peak plus 5 seconds
        assert not sae

        # testing the index error route
        peak = 200
        es, sae = d._get_start_still(0.1, time, array([150]), array([150]), peak)

        assert es == peak + 50
        assert not sae

    def test__get_transition_start(self):
        d = Detector(stillness_constraint=True)

        start = d._get_transfer_start(None, None, 50, None, None)
        assert start == 50

        d = Detector(stillness_constraint=False)

        pos_zc = array([240])
        stops = array([237])
        start = d._get_transfer_start(0.1, 250, None, pos_zc, stops)

        assert start == 237  # stillness stop within [-0.5, 0.7] seconds of ZC

        stops = array([232])
        start = d._get_transfer_start(0.1, 250, None, pos_zc, stops)

        assert start == 240  # stillness stop too far away from ZC

        # index error
        start = d._get_transfer_start(0.1, 250, None, array([260]), array([230]))

        assert start is None

    def test__is_transfer_valid(self):
        d = Detector(thresholds={"duration factor": 4, "stand displacement": 0.125})

        res = {"STS Start": []}
        peak = 350
        time = arange(0, 10, 0.01)
        vp = 3 * time
        sts_start = 300
        sts_end = 525
        prev_int_start = 0

        valid, tsi, tei = d._is_transfer_valid(
            res, peak, time, vp, sts_start, sts_end, prev_int_start
        )

        assert valid
        assert tsi == sts_start
        assert tei == sts_end

        # with existing transition
        res = {"STS Start": [2.7]}  # less than 0.4 away from time[300] = 3.0s

        valid, *_ = d._is_transfer_valid(
            res, peak, time, vp, sts_start, sts_end, prev_int_start
        )
        assert not valid
