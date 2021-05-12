from numpy import isclose, allclose

from skimu.activity.cutpoints import (
    get_available_cutpoints,
    get_level_thresholds,
    _base_cutpoints,
)


def test_get_level_thresholds():
    cuts = _base_cutpoints["migueles_wrist_adult"]

    sed_range = get_level_thresholds("sed", cuts)
    light_range = get_level_thresholds("light", cuts)
    mod_range = get_level_thresholds("mod", cuts)
    vig_range = get_level_thresholds("vig", cuts)

    assert sed_range[0] < 0.0
    assert isclose(sed_range[1], 0.050)

    assert allclose(light_range, (0.050, 0.110))

    assert allclose(mod_range, (0.110, 0.440))

    assert isclose(vig_range[0], 0.440)
    assert vig_range[1] > 16  # 16g is a large cutoff for accels


def test_get_available_cutpoints(capsys):
    get_available_cutpoints(name=None)

    out = capsys.readouterr().out

    assert True
