from numpy import allclose, sort, abs, mean, cos, pi, arange, tile, maximum, sqrt, concatenate

from skdh.activity.metrics import (
    metric_anglez,
    metric_en,
    metric_enmo,
    metric_bfen,
    metric_hfen,
    metric_hfenplus,
    metric_mad,
    metric_enmo_met,
)


def test_metric_anglez(get_linear_accel):
    x = get_linear_accel(scale=0.0).T

    res = metric_anglez(x, wlen=100)

    assert allclose(res, 90.0)


def test_metric_en(get_linear_accel):
    x = get_linear_accel(scale=0.1).T

    res = metric_en(x, wlen=100)

    assert res.shape == (5,)
    assert all(res > 0)
    assert allclose(res, 1.0, atol=0.2)


def test_metric_enmo(get_linear_accel):
    x = get_linear_accel(scale=0.5).T

    res = metric_enmo(x, wlen=100, take_abs=False, trim_zero=True)

    assert res.shape == (5,)
    assert all(res >= 0.0)

    res = metric_enmo(x, wlen=100, take_abs=True, trim_zero=True)

    assert all(res >= 0.0)

    # make sure we can get values that will actually be < 0
    x = get_linear_accel(scale=0.1).T
    res = metric_enmo(sort(x, axis=1), wlen=10, take_abs=False, trim_zero=False)

    assert any(res < 0.0)


def test_metric_bfen(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal([0.5, 1.0], [0.5, 5.0], scale=0.0)

    y[:, 2] += x

    res = metric_bfen(y, wlen=10, fs=fs, low_cutoff=1.0, high_cutoff=15.0, trim_zero=False)

    z = abs(get_sin_signal(1.0, 5.0, 0.0)[1]).reshape((-1, 10))
    z_mm = mean(z, axis=1)

    # there are some pretty large edge effects, especially on right side
    assert allclose(res[:35], z_mm[:35], atol=0.05)


def test_metric_hfen(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal([0.5, 1.0], [0.5, 5.0], scale=0.0)

    y[:, 2] += x

    res = metric_hfen(y, wlen=10, fs=fs, low_cutoff=1.0, trim_zero=False)

    z = abs(get_sin_signal(1.0, 5.0, 0.0)[1]).reshape((-1, 10))
    z_mm = mean(z, axis=1)

    # there are some pretty large edge effects, especially on right side
    assert allclose(res[7:40], z_mm[7:40], atol=0.05)


def test_metric_hfenplus(get_sin_signal, get_linear_accel):
    y = get_linear_accel(scale=0.0).T
    fs, x = get_sin_signal([0.5, 1.0], [0.5, 5.0], scale=0.0)

    y[:, 2] += x

    res = metric_hfenplus(y, wlen=10, fs=fs, cutoff=1.0, trim_zero=False)
    res_tz = metric_hfenplus(y, wlen=10, fs=fs, cutoff=1.0, trim_zero=True)

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
    res = metric_mad(y, wlen=10)

    # scale down to account for sampling frequency-ish factor
    # also make the window offset a bit to account for the double moving mean
    truth = abs(1.5 * cos(2 * pi * 0.2 * arange(0.05, 5.05, 0.1)) / 100)
    assert allclose(res, truth, atol=1e-3)


def test_metric_enmo_met(get_linear_accel):
    # 500 samples at 100 Hz, tiled to 1 minute, with norm of exactly 1g
    x = tile(get_linear_accel(scale=0.0).T, (12, 1))
    fs = 100.0

    # one minute each at 0, 50, 100, 200, and 400 mg ENMO
    y = concatenate([x * (1 + enmo / 1000) for enmo in [0.0, 50.0, 100.0, 200.0, 400.0]])

    res = metric_enmo_met(y, fs=fs)

    # net METs from White et al., with ENMO in mg and 1 MET = 71.225 J/min/kg
    assert res.shape == (5,)
    assert allclose(res, [-0.14854, 0.90726, 1.75037, 3.24089, 5.62236], atol=1e-4)


def test_metric_enmo_met_sin(get_sin_signal, get_linear_accel):
    fs, x = get_sin_signal(0.5, 1.0, scale=0.0)
    y = get_linear_accel(scale=0.0).T

    y[:, 2] += x
    # tile to 1 minute
    y = tile(y, (12, 1))

    res = metric_enmo_met(y, fs=fs)

    # 1 Hz signal so each 1s window has the same ENMO, in mg
    enmo = mean(maximum(x[:100], 0.0)) * 1000
    aee = -10.58 + 1.1176 * enmo + 2.9418 * sqrt(enmo) - 0.00059277 * enmo**2

    assert res.shape == (1,)
    assert allclose(res, aee / 71.225)
