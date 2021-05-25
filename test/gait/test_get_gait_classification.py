from skimu.gait.get_gait_classification import get_gait_classification_lgbm


class Test_get_gait_classification_lgbm:
    def test_50hz(self, gait_classification_input_50):
        t, acc = gait_classification_input_50

        starts, stops = get_gait_classification_lgbm(None, None, acc, 50.)

        assert True
