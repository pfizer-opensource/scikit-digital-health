from numpy import allclose, isclose, zeros, arange, sin, pi, nan, array, sqrt, isnan

from skimu.gait.gait_endpoints.gait_endpoints import (
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


def test_StrideTime(dummy_gait):
    st = StrideTime()

    st.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait['PARAM:stride time'],
        [2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan],
        equal_nan=True
    )


def test_StanceTime(dummy_gait):
    st = StanceTime()

    st.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait['PARAM:stance time'],
        [1.2, 1.3, 1.2, 1.3, 1.2, 1.3, 1.3, 1.1]
    )


def test_SwingTime(dummy_gait):
    st = SwingTime()

    st.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait["PARAM:swing time"],
        [0.8, nan, nan, 0.7, 0.8, 0.7, nan, nan],
        equal_nan=True
    )


def test_StepTime(dummy_gait):
    st = StepTime()
    st.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait['PARAM:step time'],
        [1.0, 1.0, nan, 1., 1., 1., 1., nan],
        equal_nan=True
    )


def test_InitialDoubleSupport(dummy_gait):
    ids = InitialDoubleSupport()
    ids.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait["PARAM:initial double support"],
        [0.3, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.1]
    )


def test_TerminalDoubleSupport(dummy_gait):
    tds = TerminalDoubleSupport()
    tds.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait["PARAM:terminal double support"],
        [0.2, 0.3, nan, 0.2, 0.3, 0.2, 0.1, nan],
        equal_nan=True
    )


def test_DoubleSupport(dummy_gait):
    ds = DoubleSupport()
    ds.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait['PARAM:double support'],
        [0.5, 0.5, nan, 0.4, 0.5, 0.5, 0.3, nan],
        equal_nan=True
    )


def test_SingleSupport(dummy_gait):
    ss = SingleSupport()
    ss.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait['PARAM:single support'],
        [0.7, 0.8, nan, 0.8, 0.8, 0.7, 0.8, nan],
        equal_nan=True
    )


def test_StepLength(dummy_gait):
    sl = StepLength()
    sl.predict(10., 1.8, dummy_gait, {})

    exp = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    exp -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    exp = 2 * sqrt(exp)
    # get predicted values and reset dictionary for another test
    pred = dummy_gait.pop("PARAM:step length")
    assert allclose(pred, exp)

    # test with no leg length provided
    sl.predict(10., None, dummy_gait, {})

    assert isnan(dummy_gait['PARAM:step length']).all()


def test_StrideLength(dummy_gait):
    sl = StrideLength()
    sl.predict(10., 1.8, dummy_gait, {})

    a = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    a -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    a = 2 * sqrt(a)

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    # get predicted values and reset dictionary for another test
    pred = dummy_gait.pop("PARAM:stride length")

    assert allclose(pred, exp, equal_nan=True)

    # test with no leg length provided
    sl.predict(10., None, dummy_gait, {})

    assert isnan(dummy_gait['PARAM:stride length']).all()


def test_GaitSpeed(dummy_gait):
    a = 2 * 1.8 * array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    a -= array([0.01, 0.04, 0.01, 0.04, 0.04, 0.04, 0.01, 0.01])
    a = 2 * sqrt(a)

    exp = a
    exp[0:2] += exp[1:3]
    exp[3:-1] += exp[4:]
    exp[[2, 7]] = nan
    exp /= array([2.0, nan, nan, 2.0, 2.0, 2.0, nan, nan])

    gs = GaitSpeed()
    gs.predict(10., 1.8, dummy_gait, {})
    pred = dummy_gait.pop('PARAM:gait speed')

    assert allclose(pred, exp, equal_nan=True)

    gs.predict(10., None, dummy_gait, {})
    assert isnan(dummy_gait['PARAM:gait speed']).all()


def test_Cadence(dummy_gait):
    c = Cadence()
    c.predict(10., 1.8, dummy_gait, {})

    assert allclose(
        dummy_gait["PARAM:cadence"],
        [60., 60., nan, 60., 60., 60., 60., nan],
        equal_nan=True
    )
