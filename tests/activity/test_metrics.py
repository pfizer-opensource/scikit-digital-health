from numpy import allclose, sort, abs, mean, cos, pi, arange

from skdh.activity.metrics import (
    metric_anglez,
    metric_en,
    metric_enmo,
    metric_bfen,
    metric_hfen,
    metric_hfenplus,
    metric_mad,
)


def test_metric_anglez(get_linear_accel):
    x = get_linear_accel(scale=0.0).T

    res = metric_anglez(x, 100)

    assert allclose(res, 90.0)


def test_metric_en(get_linear_accel):
    x = get_linear_accel(scale=0.1).T

    res = metric_en(x, 100)

    assert res.shape == (5,)
    assert all(res > 0)
    assert allclose(res, 1.0, atol=0.2)


def test_metric_enmo(get_linear_accel):
    x = get_linear_accel(scale=0.5).T

    res = metric_enmo(x, 100, take_abs=False, trim_zero=True)

    assert res.shape == (5,)
    assert all(res >= 0.0)

    res = metric_enmo(x, 100, take_abs=True, trim_zero=True)

    assert all(res >= 0.0)

    # make sure we can get values that will actually be < 0
    x = get_linear_accel(scale=0.1).T
    res = metric_enmo(sort(x, axis=1), 10, take_abs=False, trim_zero=False)

    assert any(res < 0.0)


def test_metric_bfen(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal([0.5, 1.0], [0.5, 5.0], scale=0.0)

    y[:, 2] += x

    res = metric_bfen(y, 10, fs, low_cutoff=1.0, high_cutoff=15.0, trim_zero=False)

    z = abs(get_sin_signal(1.0, 5.0, 0.0)[1]).reshape((-1, 10))
    z_mm = mean(z, axis=1)

    # there are some pretty large edge effects, especially on right side
    assert allclose(res[:35], z_mm[:35], atol=0.05)


def test_metric_hfen(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal([0.5, 1.0], [0.5, 5.0], scale=0.0)

    y[:, 2] += x

    res = metric_hfen(y, 10, fs, low_cutoff=1.0, trim_zero=False)

    z = abs(get_sin_signal(1.0, 5.0, 0.0)[1]).reshape((-1, 10))
    z_mm = mean(z, axis=1)

    # there are some pretty large edge effects, especially on right side
    assert allclose(res[7:40], z_mm[7:40], atol=0.05)


def test_metric_hfenplus(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal([0.5, 1.0], [0.5, 5.0], scale=0.0)

    y[:, 2] += x

    res = metric_hfenplus(y, 10, fs, cutoff=1.0, trim_zero=False)
    res_tz = metric_hfenplus(y, 10, fs, cutoff=1.0, trim_zero=True)

    z1 = abs(get_sin_signal(1.0, 5.0, 0.0)[1]).reshape((-1, 10))
    # no abs because the it would be abs(sin + 1) - 1 == sin_signal
    z2 = get_sin_signal(0.5, 0.5, 0.0)[1].reshape((-1, 10))

    c = z1 + z2
    c[c < 0] = 0.0

    # there are some edge effects
    assert allclose(res[6:40], mean(z1 + z2, axis=1)[6:40], atol=0.05)
    assert allclose(res_tz[6:40], mean(c, axis=1)[6:40], atol=0.05)


def test_metric_mad(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal(0.5, 0.2, scale=0.0)

    y[:, 2] += x

    # moving mean of the signal results in the deviation
    # taking the moving mean of the abs of the deviation again results
    # in something very similar to a cos wave
    res = metric_mad(y, 10)

    # scale down to account for sampling frequency-ish factor
    # also make the window offset a bit to account for the double moving mean
    truth = abs(1.5 * cos(2 * pi * 0.2 * arange(0.05, 5.05, 0.1)) / 100)
    assert allclose(res, truth, atol=1e-3)
