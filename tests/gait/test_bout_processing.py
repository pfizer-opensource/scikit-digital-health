from numpy import allclose, arange, sin, pi, array
from scipy.signal import butter, sosfiltfilt, detrend

from skdh.gait.substeps import (
    PreprocessGaitBout,
    VerticalCwtGaitEvents,
    ApCwtGaitEvents,
    CreateStridesAndQc,
    TurnDetection,
)


def test_preprocessing(t_and_x):
    t, x = t_and_x

    proc = PreprocessGaitBout(
        correct_orientation=True, filter_cutoff=20.0, filter_order=4
    )
    res = proc.predict(time=t, accel=x, fs=50.0)

    assert res["v_axis"] == 0
    assert res["ap_axis"] == 1
    assert 0.915 < res["mean_step_freq"] < 0.985


def test_vertical_cwt_gait_events(t_and_x):
    t, x = t_and_x

    # do some filtering so that it matches original implementation
    x = detrend(x, axis=0)
    sos = butter(4, 2 * 20.0 / 50.0, output="sos")
    x = sosfiltfilt(sos, x, axis=0)

    proc = VerticalCwtGaitEvents()
    res = proc.predict(
        time=t, accel_filt=x, fs=50.0, v_axis=0, v_axis_sign=1, mean_step_freq=1.0
    )

    assert allclose(
        res["initial_contacts"], [13, 63, 113, 163, 213]
    )  # peaks in the sine wave
    assert allclose(
        res["final_contacts"], [24, 76, 126, 176, 228]
    )  # peaks in the sine derivative


def test_ap_cwt_gait_events(t_and_x):
    t, x = t_and_x

    proc = ApCwtGaitEvents(
        ic_prom_factor=0.1,
        ic_dist_factor=0.0,
        fc_prom_factor=0.1,
        fc_dist_factor=0.0,
    )
    res = proc.predict(
        time=t,
        accel_filt=x,
        fs=50.0,
        v_axis=0,
        v_axis_sign=1,
        mean_step_freq=1.0,
        ap_axis=1,
        ap_axis_sign=1,
    )

    assert allclose(res["initial_contacts"], [20, 72, 125, 178])
    assert allclose(res["final_contacts"], [40, 92, 145, 197])


def test_strides_qc_static():
    t = arange(0, 10, 0.02)

    ic = array([10, 60, 100, 160, 210, 260, 305, 360])
    fc = array([15, 67, 104, 166, 218, 265, 310, 367])

    proc = CreateStridesAndQc(max_stride_time=2.25, loading_factor=0.2)
    res = proc.predict(
        time=t, fs=50.0, initial_contacts=ic, final_contacts=fc, mean_step_freq=0.95
    )

    assert allclose(res["qc_initial_contacts"], ic[:-1])
    assert allclose(res["qc_final_contacts"], fc[1:])
    assert allclose(res["forward_cycles"], [2, 2, 2, 2, 2, 1, 0])

    ic = array([10, 60, 100, 210, 260, 305, 360])
    fc = array([15, 104, 166, 218, 265, 310, 367])

    res2 = proc.predict(
        time=t, fs=50.0, initial_contacts=ic, final_contacts=fc, mean_step_freq=0.95
    )

    assert allclose(res2["qc_initial_contacts"], [100, 210, 260, 305])
    assert allclose(res2["qc_final_contacts"], [166, 265, 310, 367])
    assert allclose(res2["forward_cycles"], [0, 2, 1, 0])

    ic = array([10, 60, 100])
    fc = array([44, 67, 104])

    res3 = proc.predict(
        time=t, fs=50.0, initial_contacts=ic, final_contacts=fc, mean_step_freq=0.95
    )

    assert allclose(res3["qc_initial_contacts"], [60])
    assert allclose(res3["qc_final_contacts"], [104])
    assert allclose(res3["forward_cycles"], [0])


def test_turns(gait_input_gyro):
    t, acc, gyro = gait_input_gyro

    # trim a bit, don't need all the data
    n = t.size
    n2 = int(n / 2)
    t = t[:n2]
    acc = acc[:n2]
    gyro = gyro[:n2]

    # range of simulated IC/FC
    ic = arange(50, int(0.9 * acc.shape[0]), 55)[:-1]
    fc = arange(109, int(0.9 * acc.shape[0]), 55)

    proc = TurnDetection()
    res = proc.predict(
        time=t,
        accel=acc,
        gyro=gyro,
        fs=50.0,
        qc_initial_contacts=ic,
        qc_final_contacts=fc,
    )

    truth = array(
        [0] * 17 + [1, 2, 2, 2, 2, 2, 2] + [0] * 8 + [1, 2, 2, 2, 2] + [0] * 8
    )

    assert allclose(res["step_in_turn"], truth)
