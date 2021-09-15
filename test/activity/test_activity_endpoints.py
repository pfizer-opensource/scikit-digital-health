from numpy import isclose

from skdh.activity.endpoints import get_activity_bouts, ActivityEndpoint, IntensityGradient, MaxAcceleration, TotalIntensityTime, BoutIntensityTime, FragmentationEndpoints


def test_get_activity_bouts(act_acc):
    res_1_nclosed = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 1)
    res_1_closed = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, False, 1)
    res_2 = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 2)
    res_3 = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 3)
    res_4 = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 4)

    true_1_nclosed = 21.
    true_1_closed = 20.
    true_2 = 24.
    true_3 = 22.
    true_4 = 22.

    assert isclose(res_1_nclosed, true_1_nclosed)
    assert isclose(res_1_closed, true_1_closed)
    assert isclose(res_2, true_2)
    assert isclose(res_3, true_3)
    assert isclose(res_4, true_4)
