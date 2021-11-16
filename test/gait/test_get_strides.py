from numpy import allclose, array, arange, pi, sin, sum, unique

from skdh.gait.get_strides import get_strides


def test_get_strides():
    t = arange(0, 10, 0.02)
    x = 1 + 0.75 * sin(2 * pi * 1.0 * t)

    ic = array([10, 60, 100, 160, 210, 260, 305, 360])
    fc = array([15, 67, 104, 166, 218, 265, 310, 367])

    gait = {
        i: []
        for i in ["IC", "FC", "FC opp foot", "forward cycles", "delta h", "IC Time"]
    }

    n_steps = get_strides(gait, x, 0, ic, fc, t, 50.0, 2.25, 0.2)

    assert n_steps == 7
    assert allclose(gait["IC"], ic[:n_steps])
    assert allclose(gait["FC"], fc[1:])
    assert unique(gait["FC"]).size == array(gait["FC"]).size
    assert allclose(gait["FC opp foot"], fc[:n_steps])
    assert unique(gait["FC opp foot"]).size == array(gait["FC opp foot"]).size
    assert sum(array(gait["forward cycles"]) > 1) == 5
    assert allclose(array(gait["delta h"])[[0, 3, 4]], 5.073504651555079)

    # second set to catch some continues/nan filling
    ic = array([10, 60, 100])
    fc = array([44, 67, 104])

    gait = {
        i: []
        for i in ["IC", "FC", "FC opp foot", "forward cycles", "delta h", "IC Time"]
    }

    n_steps = get_strides(gait, x, 0, ic, fc, t, 50.0, 2.25, 0.2)

    assert n_steps == 1
    assert allclose(gait["forward cycles"], 0)
