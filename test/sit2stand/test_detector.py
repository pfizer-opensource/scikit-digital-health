from numpy import arange, allclose, sqrt

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
