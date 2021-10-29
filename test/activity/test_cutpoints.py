import pytest
from numpy import isclose, allclose

from skdh.activity.cutpoints import (
    get_metric,
    get_available_cutpoints,
    get_level_thresholds,
    _base_cutpoints,
)


def test_get_metric():
    f = get_metric("metric_enmo")
    assert "metric_enmo" in str(f)

    f = get_metric("metric_mad")
    assert "metric_mad" in str(f)


def test_get_level_thresholds():
    cuts = _base_cutpoints["migueles_wrist_adult"]

    sed_range = get_level_thresholds("sed", cuts)
    light_range = get_level_thresholds("light", cuts)
    mod_range = get_level_thresholds("mod", cuts)
    vig_range = get_level_thresholds("vig", cuts)
    mvpa_range = get_level_thresholds("mvpa", cuts)

    assert sed_range[0] < 0.0
    assert isclose(sed_range[1], 0.050)

    assert allclose(light_range, (0.050, 0.110))

    assert allclose(mod_range, (0.110, 0.440))

    assert isclose(vig_range[0], 0.440)
    assert vig_range[1] > 16  # 16g is a large cutoff for accels

    assert isclose(mvpa_range[0], 0.110)
    assert mvpa_range[1] > 16

    with pytest.raises(ValueError):
        get_level_thresholds("bad level", cuts)


def test_get_available_cutpoints(capsys):
    get_available_cutpoints(name=None)

    out = capsys.readouterr().out
    # check that a few of the values are all present
    to_check = [
        "esliger_lwrist_adult",
        "phillips_lwrist_child8-14",
        "vaha-ypya_hip_adult",
        "hildebrand_wrist_adult_geneactiv",
        "migueles_wrist_adult",
    ]
    assert all([f"{i}\n" in out for i in to_check])

    get_available_cutpoints("migueles_wrist_adult")

    out2 = capsys.readouterr().out

    assert "metric_enmo" in out2
    assert "sedentary range" in out2
    assert "light range" in out2
    assert "moderate range" in out2
    assert "vigorous range" in out2
