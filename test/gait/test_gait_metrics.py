import pytest
from numpy import allclose, isclose, zeros, arange, sin, pi, nan, array, sqrt, isnan

from skdh.gait.gait_metrics.gait_metrics import (
    _autocovariancefn,
    StrideTime,
    StanceTime,
    SwingTime,
    StepTime,
    InitialDoubleSupport,
    TerminalDoubleSupport,
    DoubleSupport,
    SingleSupport,
    StepLengthModel2,
    StrideLengthModel2,
    GaitSpeedModel2,
    Cadence,
    StepLengthModel1,
    StrideLengthModel1,
    GaitSpeedModel1,
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

    st.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["stride time"],
        [2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan],
        equal_nan=True,
    )


def test_StanceTime(d_gait):
    st = StanceTime()

    st.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(d_gait["stance time"], [1.2, 1.3, 1.2, 1.3, 1.2, 1.3, 1.3, 1.1])


def test_SwingTime(d_gait):
    st = SwingTime()

    st.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["swing time"],
        [0.8, nan, nan, 0.7, 0.8, 0.7, nan, nan],
        equal_nan=True,
    )


def test_StepTime(d_gait):
    st = StepTime()
    st.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["step time"],
        [1.0, 1.0, nan, 1.0, 1.0, 1.0, 1.0, nan],
        equal_nan=True,
    )


def test_InitialDoubleSupport(d_gait):
    ids = InitialDoubleSupport()
    ids.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["initial double support"], [0.3, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.1]
    )


def test_TerminalDoubleSupport(d_gait):
    tds = TerminalDoubleSupport()
    tds.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["terminal double support"],
        [0.2, 0.3, nan, 0.2, 0.3, 0.2, 0.1, nan],
        equal_nan=True,
    )


def test_DoubleSupport(d_gait):
    ds = DoubleSupport()
    ds.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["double support"],
        [0.5, 0.5, nan, 0.4, 0.5, 0.5, 0.3, nan],
        equal_nan=True,
    )


def test_SingleSupport(d_gait):
    ss = SingleSupport()
    ss.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["single support"],
        [0.7, 0.8, nan, 0.8, 0.8, 0.7, 0.8, nan],
        equal_nan=True,
    )


def test_StepLengthModel1(d_gait, d_gait_aux):
    sl = StepLengthModel1()
    sl.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux=d_gait_aux)

    exp = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    exp -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    exp = 2 * sqrt(exp)
    # get predicted values and reset dictionary for another test
    pred = d_gait.pop("step length m1")
    assert allclose(pred, exp)

    # test with no leg length provided
    sl.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert isnan(d_gait["step length m1"]).all()


def test_StepLengthModel2(d_gait, d_gait_aux):
    sl = StepLengthModel2()
    sl.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux=d_gait_aux)

    lprime = array([0.792, 0.792, 0.792, 0.495, 0.495, 0.495, 0.495, 0.495])

    exp = 2 * 1.8 * d_gait["m2 delta h"]
    exp -= d_gait["m2 delta h"] ** 2
    exp = 2 * sqrt(exp)
    exp += 2 * sqrt(
        2 * lprime * d_gait["m2 delta h prime"] - d_gait["m2 delta h prime"] ** 2
    )

    pred = d_gait.pop("step length")

    assert allclose(pred, exp)

    # test with no leg length provided
    sl.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert isnan(d_gait["step length"]).all()


def test_StrideLengthModel1(d_gait, d_gait_aux):
    sl = StrideLengthModel1()
    sl.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux=d_gait_aux)

    a = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    a -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    a = 2 * sqrt(a)

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    # get predicted values and reset dictionary for another test
    pred = d_gait.pop("stride length m1")

    assert allclose(pred, exp, equal_nan=True)

    # test with no leg length provided
    sl.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert isnan(d_gait["stride length m1"]).all()


def test_StrideLengthModel2(d_gait, d_gait_aux):
    sl = StrideLengthModel2()
    sl.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux=d_gait_aux)

    lprime = array([0.792, 0.792, 0.792, 0.495, 0.495, 0.495, 0.495, 0.495])

    a = 2 * 1.8 * d_gait["m2 delta h"]
    a -= d_gait["m2 delta h"] ** 2
    a = 2 * sqrt(a)
    a += 2 * sqrt(
        2 * lprime * d_gait["m2 delta h prime"] - d_gait["m2 delta h prime"] ** 2
    )

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    # get predicted values and reset dictionary for another test
    pred = d_gait.pop("stride length")

    assert allclose(pred, exp, equal_nan=True)

    # test with no leg length provided
    sl.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert isnan(d_gait["stride length"]).all()


def test_GaitSpeedModel1(d_gait, d_gait_aux):
    a = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    a -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    a = 2 * sqrt(a)

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    exp /= array([2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    gs = GaitSpeedModel1()
    gs.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux=d_gait_aux)
    pred = d_gait.pop("gait speed m1")

    assert allclose(pred, exp, equal_nan=True)

    gs.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)
    assert isnan(d_gait["gait speed m1"]).all()


def test_GaitSpeedModel2(d_gait, d_gait_aux):
    lprime = array([0.792, 0.792, 0.792, 0.495, 0.495, 0.495, 0.495, 0.495])
    a = 2 * 1.8 * d_gait["m2 delta h"]
    a -= d_gait["m2 delta h"] ** 2
    a = 2 * sqrt(a)
    a += 2 * sqrt(
        2 * lprime * d_gait["m2 delta h prime"] - d_gait["m2 delta h prime"] ** 2
    )

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    exp /= array([2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    gs = GaitSpeedModel2()
    gs.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux=d_gait_aux)
    pred = d_gait.pop("gait speed")

    assert allclose(pred, exp, equal_nan=True)

    gs.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)
    assert isnan(d_gait["gait speed"]).all()


def test_Cadence(d_gait):
    c = Cadence()
    c.predict(fs=50.0, leg_length=1.8, gait=d_gait, gait_aux={})

    assert allclose(
        d_gait["cadence"],
        [60.0, 60.0, nan, 60.0, 60.0, 60.0, 60.0, nan],
        equal_nan=True,
    )


def test_IntraStrideCovarianceV(d_gait, d_gait_aux):
    iscv = IntraStrideCovarianceV()
    iscv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert allclose(
        d_gait["intra-stride covariance - V"],
        [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan],
        equal_nan=True,
    )


def test_IntraStepCovarianceV(d_gait, d_gait_aux):
    iscv = IntraStepCovarianceV()
    iscv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert allclose(
        d_gait["intra-step covariance - V"],
        [1.0, nan, nan, 1.0, 1.0, 1.0, 1.0, nan],
        equal_nan=True,
    )


def test_HarmonicRatioV(d_gait, d_gait_aux):
    hrv = HarmonicRatioV()
    hrv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)
    # get predicted values and reset d_gait for another run
    pred = d_gait.pop("harmonic ratio - V")

    # values are somewhat weird because stride frequency is different than
    # the frequency used to create the "acceleration" data
    assert allclose(
        pred,
        [2.54304311, nan, nan, 2.54304311, 2.54304311, 2.54304311, nan, nan],
        equal_nan=True,
    )

    # <= 10 harmonics
    hrv.predict(fs=10.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)
    pred = d_gait.pop("harmonic ratio - V")

    assert isnan(pred).all()

    # test with less than 20 harmonics
    hrv.predict(fs=15.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)
    pred = d_gait.pop("harmonic ratio - V")

    assert allclose(
        pred,
        [0.65066714, nan, nan, 0.65066714, 0.65066714, 0.65066714, nan, nan],
        equal_nan=True,
    )


def test_StrideSPARC(d_gait, d_gait_aux):
    ss = StrideSPARC()
    ss.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert allclose(
        d_gait["stride SPARC"],
        [-3.27853883, nan, nan, -3.27853883, -3.27853883, -3.27853883, nan, nan],
        equal_nan=True,
    )


def test_PhaseCoordinationIndex(d_gait, d_gait_aux):
    pci = PhaseCoordinationIndex()
    with pytest.warns(RuntimeWarning):
        pci.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    # 0 values since the phase mean - 0.5 is 0.0
    assert allclose(
        d_gait["bout:phase coordination index"],
        [nan, nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        equal_nan=True,
    )


def test_GaitSymmetryIndex(d_gait, d_gait_aux):
    # set manually so one of the bouts doesnt have a median step time
    d_gait["stride time"] = array([nan, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    gsi = GaitSymmetryIndex()
    with pytest.warns(RuntimeWarning):  # all nan slice
        gsi.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert allclose(
        d_gait["bout:gait symmetry index"],
        [nan, nan, nan, 0.97467939, 0.97467939, 0.97467939, 0.97467939, 0.97467939],
        equal_nan=True,
    )


def test_StepRegularityV(d_gait, d_gait_aux):
    # set manually so one of the bouts doesnt have a median step time
    d_gait["step time"] = array([nan, nan, nan, 1.0, 1.0, 1.0, 1.0, nan])

    srv = StepRegularityV()
    with pytest.warns(RuntimeWarning):  # all nan slice
        srv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert allclose(
        d_gait["bout:step regularity - V"],
        [nan, nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        equal_nan=True,
    )


def test_StrideRegularityV(d_gait, d_gait_aux):
    # set manually so one of the bouts doesnt have a median step time
    d_gait["stride time"] = array([nan, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    srv = StrideRegularityV()
    with pytest.warns(RuntimeWarning):  # all nan slice
        srv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    assert allclose(
        d_gait["bout:stride regularity - V"],
        [nan, nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        equal_nan=True,
    )


def test_AutocovarianceSymmetryV(d_gait, d_gait_aux):
    acsv = AutocovarianceSymmetryV()
    acsv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    # simple difference between stride and step regularity: 1 - 1 = 0
    assert allclose(d_gait["bout:autocovariance symmetry - V"], 0.0)


def test_RegularityIndexV(d_gait, d_gait_aux):
    riv = RegularityIndexV()
    riv.predict(fs=50.0, leg_length=None, gait=d_gait, gait_aux=d_gait_aux)

    # 1 - 0 = 1
    assert allclose(d_gait["bout:regularity index - V"], 1.0)
