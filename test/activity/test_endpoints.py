from numpy import isclose

from skdh.activity.endpoints import get_activity_bouts, ActivityEndpoint, IntensityGradient, MaxAcceleration, TotalIntensityTime, BoutIntensityTime, FragmentationEndpoints


class TestGetActivityBouts:
    def test_boutmetric_1(self, act_acc):
        res_a = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 1)
        res_b = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, False, 1)

        true_a = 7. + 8. + 6.
        true_b = 7. + 7. + 6.

        assert isclose(res_a, true_a)
        assert isclose(res_b, true_b)

    def test_boutmetric_2(self, act_acc):
        res_a = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 2)
        true_a = 24.

        assert isclose(res_a, true_a)

    def test_boutmetric_3(self, act_acc):
        res_a = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 3)
        true_a = 22.

        assert isclose(res_a, true_a)

    def test_boutmetric_4(self, act_acc):
        res_a = get_activity_bouts(act_acc, 0.5, 1.5, 60, 6, 0.8, True, 4)
        true_a = 22.

        assert isclose(res_a, true_a)
