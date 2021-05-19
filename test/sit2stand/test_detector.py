import pytest
from numpy import array, allclose, arange

from skimu.sit2stand.detector import pad_moving_sd, get_stillness, Detector


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
    still, starts, stops = get_stillness(
        dummy_stillness_data * 9.81,
        1/20,
        9.81,
        0.25,
        threshs,
    )

    assert allclose(starts, 300 + 3)  # account for padding
    assert allclose(stops, 400 - 3)  # account for padding
    assert still.sum() == 100 - 3 - 3  # padding


class TestDetector:
    def test_update_thresh(self):
        d = Detector(thresholds={"accel moving avg": 0.5, "test": 100})

        assert d.thresh["accel moving avg"] == 0.5

        for k in d._default_thresholds:
            if k != "accel moving avg":
                assert d.thresh[k] == d._default_thresholds[k]

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

        es, sae = d._get_start_still(time, array([125, 350]), array([125]), 100)

        assert es == 125
        assert sae

        time = arange(0, 40, 0.1)
        with pytest.raises(IndexError):
            es, sae = d._get_start_still(time, array([370]), array([370]), 50)

