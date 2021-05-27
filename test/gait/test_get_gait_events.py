from numpy import isclose, allclose, arange, sin, pi, zeros

from skdh.gait.get_gait_events import get_cwt_scales, get_gait_events


def test_get_cwt_scales():
    s1, s2 = get_cwt_scales(False, None, 8, 50.0)
    assert s1 == s2 == 8

    # optimal scaling
    t = arange(0, 10, 0.02)
    x = sin(2 * pi * 1.0 * t)

    scale1, scale2 = get_cwt_scales(True, x, 8, 50.0)

    # IC scale: -10 * sf + 56
    # FC scale: -52 * sf + 131
    # at 250hz, so scale by 5
    assert isclose(scale1, round(46 / 5))
    assert isclose(scale2, round(79 / 5))


def test_get_gait_events():
    t = arange(0, 5.01, 0.02)
    x = zeros((t.size, 3))
    x[:, 0] += 1 + 0.75 * sin(2 * pi * 1.0 * t)
    x[:, 1] += 0.3 * sin(2 * pi * 1.0 * t)
    x[:, 2] += 0.1 * sin(2 * pi * 2.0 * t)

    ic, fc, fva, va = get_gait_events(x, 50.0, t, 8, 4, 20.0, True, True)

    assert va == 0
    assert allclose(ic, [13, 63, 113, 163, 213])  # peaks in the sine wave
    assert allclose(fc, [24, 76, 126, 176, 228])  # peaks in the sine derivative
