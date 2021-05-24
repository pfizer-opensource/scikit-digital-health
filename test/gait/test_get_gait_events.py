from numpy import isclose, arange, sin, pi

from skimu.gait.get_gait_events import get_cwt_scales, get_gait_events


def test_get_cwt_scales():
    s1, s2 = get_cwt_scales(False, None, 8, 50.)
    assert s1 == s2 == 8

    # optimal scaling
    t = arange(0, 10, 0.02)
    x = sin(2 * pi * 1.0 * t)

    scale1, scale2 = get_cwt_scales(True, x, 8, 50.)

    # IC scale: -10 * sf + 56
    # FC scale: -52 * sf + 131
    # at 250hz, so scale by 5
    assert isclose(scale1, round(46 / 5))
    assert isclose(scale2, round(79 / 5))
