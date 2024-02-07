import pytest
import numpy as np

from skdh.utility import fragmentation_endpoints as fe
from skdh.utility.internal import rle


class TestGini:
    """
    Test values pulled from here:
    https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-
    python/48999797#48999797
    """

    def test1(self):
        x = np.array([1, 1, 1, 1, 1000])

        assert np.isclose(np.around(fe.gini(x, corr=False), 3), 0.796)
        assert np.isclose(np.around(fe.gini(x, corr=True), 3), 0.995)

    def test2(self):
        x = np.array([3, 1, 6, 2, 1])
        w = np.array([4, 2, 2, 10, 1])

        assert np.isclose(np.around(fe.gini(x, w=w, corr=False), 4), 0.2553)
        assert np.isclose(
            np.around(fe.gini(x, w=w, corr=True), 4), np.around(0.2553 * 5 / 4, 4)
        )

    def test_size(self):
        x = fe.gini(np.array([]), None, True)
        assert np.isclose(x, 0.0)


class _BaseTestFragEndpoint:
    def test(self, dummy_frag_predictions):
        l, s, v = rle(dummy_frag_predictions)

        res_0 = self.metric(dummy_frag_predictions, voi=0)
        res_1 = self.metric(dummy_frag_predictions, voi=1)

        assert np.isclose(res_0, self.normal_result_0, equal_nan=True)
        assert np.isclose(res_1, self.normal_result_1, equal_nan=True)

        res_0 = self.metric(lengths=l, values=v, voi=0)
        res_1 = self.metric(lengths=l, values=v, voi=1)

        assert np.isclose(res_0, self.normal_result_0, equal_nan=True)
        assert np.isclose(res_1, self.normal_result_1, equal_nan=True)

        res_0 = self.metric(lengths=l[v == 0])
        res_1 = self.metric(lengths=l[v == 1])

        assert np.isclose(res_0, self.normal_result_0, equal_nan=True)
        assert np.isclose(res_1, self.normal_result_1, equal_nan=True)

    def test_all_zeros(self):
        pred = np.zeros(30)
        l, s, v = rle(pred)

        res_0 = self.metric(pred, voi=0)
        res_1 = self.metric(pred, voi=1)

        assert np.isclose(res_0, self.zeros_result_0, equal_nan=True)
        assert np.isclose(res_1, self.zeros_result_1, equal_nan=True)

        res_0 = self.metric(lengths=l, values=v, voi=0)
        res_1 = self.metric(lengths=l, values=v, voi=1)

        assert np.isclose(res_0, self.zeros_result_0, equal_nan=True)
        assert np.isclose(res_1, self.zeros_result_1, equal_nan=True)

        res_0 = self.metric(lengths=l[v == 0])
        res_1 = self.metric(lengths=l[v == 1])

        assert np.isclose(res_0, self.zeros_result_0, equal_nan=True)
        assert np.isclose(res_1, self.zeros_result_1, equal_nan=True)

    def test_missing_inputs(self):
        with pytest.raises(ValueError):
            self.metric(a=None, lengths=None)


class TestAverageDuration(_BaseTestFragEndpoint):
    metric = staticmethod(fe.average_duration)
    normal_result_0 = 17 / 4
    normal_result_1 = 13 / 3
    zeros_result_0 = 30.0
    zeros_result_1 = 0.0


class TestStateTransitionProbability(_BaseTestFragEndpoint):
    metric = staticmethod(fe.state_transition_probability)
    normal_result_0 = 4 / 17
    normal_result_1 = 3 / 13
    zeros_result_0 = 1 / 30
    zeros_result_1 = np.nan


class TestGiniIndex(_BaseTestFragEndpoint):
    metric = staticmethod(fe.gini_index)
    normal_result_0 = 1 / 3
    normal_result_1 = 0.3076923
    zeros_result_0 = 1.0
    zeros_result_1 = 0.0


class TestAverageHazard(_BaseTestFragEndpoint):
    metric = staticmethod(fe.average_hazard)
    normal_result_0 = 0.52083333333333
    normal_result_1 = 0.8333333
    zeros_result_0 = 1.0
    zeros_result_1 = np.nan


class TestStatePowerLawDistribution(_BaseTestFragEndpoint):
    metric = staticmethod(fe.state_power_law_distribution)
    normal_result_0 = 2.073754
    normal_result_1 = 3.151675
    zeros_result_0 = 1 + 1 / np.log(30 / 29.5)
    zeros_result_1 = 1.0
