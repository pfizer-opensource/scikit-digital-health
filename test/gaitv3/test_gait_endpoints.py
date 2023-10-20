import pytest
from numpy import allclose, isclose, zeros, arange, sin, pi, nan, array, sqrt, isnan

from skdh.gait.gait_endpoints.gait_endpoints import (
    _autocovariancefn,
    StrideTime,
    StanceTime,
    SwingTime,
    StepTime,
    InitialDoubleSupport,
    TerminalDoubleSupport,
    DoubleSupport,
    SingleSupport,
    StepLength,
    StrideLength,
    GaitSpeed,
    Cadence,
    GaitSymmetryIndex,
    IntraStepCovarianceV,
    IntraStrideCovarianceV,
    HarmonicRatioV,
    StrideSPARC,
    PhaseCoordinationIndex,
    StepRegularityV,
    StrideRegularityV,
    AutocovarianceSymmetryV,
    RegularityIndexV,
)


def test__autocovariancefn():
    t = arange(0, 10, 0.01)

    x = zeros((2, 1000))
    x[0] = sin(2 * pi * 1.0 * t)
    x[1] = sin(2 * pi * 0.5 * t)

    ac = _autocovariancefn(x, 500, biased=True, axis=1)

    assert allclose(ac[:, 0], 1.0)
    # match up every 200 samples
    assert isclose(ac[0, 200], ac[1, 200])
    assert isclose(ac[0, 400], ac[1, 400])
    # offset every 100 samples
    assert isclose(ac[0, 100], -ac[1, 100])
    assert isclose(ac[0, 300], -ac[1, 300])


def test_StrideTime(d_gait):
    st = StrideTime()

    st.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:stride time"],
        [2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan],
        equal_nan=True,
    )


def test_StanceTime(d_gait):
    st = StanceTime()

    st.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:stance time"], [1.2, 1.3, 1.2, 1.3, 1.2, 1.3, 1.3, 1.1]
    )


def test_SwingTime(d_gait):
    st = SwingTime()

    st.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:swing time"],
        [0.8, nan, nan, 0.7, 0.8, 0.7, nan, nan],
        equal_nan=True,
    )


def test_StepTime(d_gait):
    st = StepTime()
    st.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:step time"],
        [1.0, 1.0, nan, 1.0, 1.0, 1.0, 1.0, nan],
        equal_nan=True,
    )


def test_InitialDoubleSupport(d_gait):
    ids = InitialDoubleSupport()
    ids.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:initial double support"], [0.3, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.1]
    )


def test_TerminalDoubleSupport(d_gait):
    tds = TerminalDoubleSupport()
    tds.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:terminal double support"],
        [0.2, 0.3, nan, 0.2, 0.3, 0.2, 0.1, nan],
        equal_nan=True,
    )


def test_DoubleSupport(d_gait):
    ds = DoubleSupport()
    ds.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:double support"],
        [0.5, 0.5, nan, 0.4, 0.5, 0.5, 0.3, nan],
        equal_nan=True,
    )


def test_SingleSupport(d_gait):
    ss = SingleSupport()
    ss.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:single support"],
        [0.7, 0.8, nan, 0.8, 0.8, 0.7, 0.8, nan],
        equal_nan=True,
    )


def test_StepLength(d_gait):
    sl = StepLength()
    sl.predict(50.0, 1.8, d_gait, {})

    exp = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    exp -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    exp = 2 * sqrt(exp)
    # get predicted values and reset dictionary for another test
    pred = d_gait.pop("PARAM:step length")
    assert allclose(pred, exp)

    # test with no leg length provided
    sl.predict(50.0, None, d_gait, {})

    assert isnan(d_gait["PARAM:step length"]).all()


def test_StrideLength(d_gait):
    sl = StrideLength()
    sl.predict(50.0, 1.8, d_gait, {})

    a = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    a -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    a = 2 * sqrt(a)

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    # get predicted values and reset dictionary for another test
    pred = d_gait.pop("PARAM:stride length")

    assert allclose(pred, exp, equal_nan=True)

    # test with no leg length provided
    sl.predict(50.0, None, d_gait, {})

    assert isnan(d_gait["PARAM:stride length"]).all()


def test_GaitSpeed(d_gait):
    a = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    a -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    a = 2 * sqrt(a)

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    exp /= array([2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    gs = GaitSpeed()
    gs.predict(50.0, 1.8, d_gait, {})
    pred = d_gait.pop("PARAM:gait speed")

    assert allclose(pred, exp, equal_nan=True)

    gs.predict(50.0, None, d_gait, {})
    assert isnan(d_gait["PARAM:gait speed"]).all()


def test_Cadence(d_gait):
    c = Cadence()
    c.predict(50.0, 1.8, d_gait, {})

    assert allclose(
        d_gait["PARAM:cadence"],
        [60.0, 60.0, nan, 60.0, 60.0, 60.0, 60.0, nan],
        equal_nan=True,
    )


def test_IntraStrideCovarianceV(d_gait, d_gait_aux):
    iscv = IntraStrideCovarianceV()
    iscv.predict(50.0, None, d_gait, d_gait_aux)

    assert allclose(
        d_gait["PARAM:intra-stride covariance - V"],
        [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan],
        equal_nan=True,
    )


def test_IntraStepCovarianceV(d_gait, d_gait_aux):
    iscv = IntraStepCovarianceV()
    iscv.predict(50.0, None, d_gait, d_gait_aux)

    assert allclose(
        d_gait["PARAM:intra-step covariance - V"],
        [1.0, nan, nan, 1.0, 1.0, 1.0, 1.0, nan],
        equal_nan=True,
    )


def test_HarmonicRatioV(d_gait, d_gait_aux):
    hrv = HarmonicRatioV()
    hrv.predict(50.0, None, d_gait, d_gait_aux)
    # get predicted values and reset d_gait for another run
    pred = d_gait.pop("PARAM:harmonic ratio - V")

    # values are somewhat weird because stride frequency is different than
    # the frequency used to create the "acceleration" data
    assert allclose(
        pred,
        [2.54304311, nan, nan, 2.54304311, 2.54304311, 2.54304311, nan, nan],
        equal_nan=True,
    )

    # <= 10 harmonics
    hrv.predict(10.0, None, d_gait, d_gait_aux)
    pred = d_gait.pop("PARAM:harmonic ratio - V")

    assert isnan(pred).all()

    # test with less than 20 harmonics
    hrv.predict(15.0, None, d_gait, d_gait_aux)
    pred = d_gait.pop("PARAM:harmonic ratio - V")

    assert allclose(
        pred,
        [0.65066714, nan, nan, 0.65066714, 0.65066714, 0.65066714, nan, nan],
        equal_nan=True,
    )


def test_StrideSPARC(d_gait, d_gait_aux):
    ss = StrideSPARC()
    ss.predict(50.0, None, d_gait, d_gait_aux)

    assert allclose(
        d_gait["PARAM:stride SPARC"],
        [-3.27853883, nan, nan, -3.27853883, -3.27853883, -3.27853883, nan, nan],
        equal_nan=True,
    )


def test_PhaseCoordinationIndex(d_gait, d_gait_aux):
    pci = PhaseCoordinationIndex()
    with pytest.warns(RuntimeWarning):
        pci.predict(50.0, None, d_gait, d_gait_aux)

    # 0 values since the phase mean - 0.5 is 0.0
    assert allclose(
        d_gait["BOUTPARAM:phase coordination index"],
        [nan, nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        equal_nan=True,
    )


def test_GaitSymmetryIndex(d_gait, d_gait_aux):
    # set manually so one of the bouts doesnt have a median step time
    d_gait["PARAM:stride time"] = array([nan, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    gsi = GaitSymmetryIndex()
    with pytest.warns(RuntimeWarning):  # all nan slice
        gsi.predict(50.0, None, d_gait, d_gait_aux)

    assert allclose(
        d_gait["BOUTPARAM:gait symmetry index"],
        [nan, nan, nan, 0.97467939, 0.97467939, 0.97467939, 0.97467939, 0.97467939],
        equal_nan=True,
    )


def test_StepRegularityV(d_gait, d_gait_aux):
    # set manually so one of the bouts doesnt have a median step time
    d_gait["PARAM:step time"] = array([nan, nan, nan, 1.0, 1.0, 1.0, 1.0, nan])

    srv = StepRegularityV()
    with pytest.warns(RuntimeWarning):  # all nan slice
        srv.predict(50.0, None, d_gait, d_gait_aux)

    assert allclose(
        d_gait["BOUTPARAM:step regularity - V"],
        [nan, nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        equal_nan=True,
    )


def test_StrideRegularityV(d_gait, d_gait_aux):
    # set manually so one of the bouts doesnt have a median step time
    d_gait["PARAM:stride time"] = array([nan, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    srv = StrideRegularityV()
    with pytest.warns(RuntimeWarning):  # all nan slice
        srv.predict(50.0, None, d_gait, d_gait_aux)

    assert allclose(
        d_gait["BOUTPARAM:stride regularity - V"],
        [nan, nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        equal_nan=True,
    )


def test_AutocovarianceSymmetryV(d_gait, d_gait_aux):
    acsv = AutocovarianceSymmetryV()
    acsv.predict(50.0, None, d_gait, d_gait_aux)

    # simple difference between stride and step regularity: 1 - 1 = 0
    assert allclose(d_gait["BOUTPARAM:autocovariance symmetry - V"], 0.0)


def test_RegularityIndexV(d_gait, d_gait_aux):
    riv = RegularityIndexV()
    riv.predict(50.0, None, d_gait, d_gait_aux)

    # 1 - 0 = 1
    assert allclose(d_gait["BOUTPARAM:regularity index - V"], 1.0)
