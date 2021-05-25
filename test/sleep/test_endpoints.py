import numpy as np

from skdh.sleep.utility import rle
from skdh.sleep.endpoints import *


class _BaseTestEndpoint:
    def test(self, dummy_sleep_predictions):
        l, s, v = rle(dummy_sleep_predictions)

        res = self.metric().predict(
            sleep_predictions=dummy_sleep_predictions, lengths=l, starts=s, values=v
        )

        assert np.isclose(res, self.normal_result, equal_nan=True)

    def test_all_zeros(self):
        sleep_pred = np.zeros(30)
        l, s, v = rle(sleep_pred)

        res = self.metric().predict(
            sleep_predictions=sleep_pred, lengths=l, starts=s, values=v
        )

        assert np.isclose(res, self.zeros_result, equal_nan=True)

    def test_all_ones(self):
        sleep_pred = np.ones(30)
        l, s, v = rle(sleep_pred)

        res = self.metric().predict(
            sleep_predictions=sleep_pred, lengths=l, starts=s, values=v
        )

        assert np.isclose(res, self.ones_result, equal_nan=True)


class TestTotalSleepTime(_BaseTestEndpoint):
    metric = TotalSleepTime
    normal_result = 13
    zeros_result = 0
    ones_result = 30


class TestPercentTimeAsleep(_BaseTestEndpoint):
    metric = PercentTimeAsleep
    normal_result = 13 / 30 * 100
    zeros_result = 0.0
    ones_result = 100.0


class TestNumberWakeBouts(_BaseTestEndpoint):
    metric = NumberWakeBouts
    normal_result = 2
    zeros_result = 0
    ones_result = 0


class TestSleepOnsetLatency(_BaseTestEndpoint):
    metric = SleepOnsetLatency
    normal_result = 3
    zeros_result = np.nan
    ones_result = 0.0


class TestWakeAfterSleepOnset(_BaseTestEndpoint):
    metric = WakeAfterSleepOnset
    normal_result = 7
    zeros_result = np.nan
    ones_result = 0.0


class TestAverageSleepDuration(_BaseTestEndpoint):
    metric = AverageSleepDuration
    normal_result = 13 / 3
    zeros_result = 0.0
    ones_result = 30.0


class TestAverageWakeDuration(_BaseTestEndpoint):
    metric = AverageWakeDuration
    normal_result = 17 / 4
    zeros_result = 30.0
    ones_result = 0.0


class TestSleepWakeTransitionProbability(_BaseTestEndpoint):
    metric = SleepWakeTransitionProbability
    normal_result = 3 / 13
    zeros_result = np.nan
    ones_result = 1 / 30


class TestWakeSleepTransitionProbability(_BaseTestEndpoint):
    metric = WakeSleepTransitionProbability
    normal_result = 4 / 17
    zeros_result = 1 / 30
    ones_result = np.nan


class TestSleepGiniIndex(_BaseTestEndpoint):
    metric = SleepGiniIndex
    normal_result = 0.3076923
    zeros_result = 0.0
    ones_result = 1.0


class TestWakeGiniIndex(_BaseTestEndpoint):
    metric = WakeGiniIndex
    normal_result = 1 / 3
    zeros_result = 1.0
    ones_result = 0.0


class TestSleepAverageHazard(_BaseTestEndpoint):
    metric = SleepAverageHazard
    normal_result = 0.8333333
    zeros_result = np.nan
    ones_result = 1.0


class TestWakeAverageHazard(_BaseTestEndpoint):
    metric = WakeAverageHazard
    normal_result = 0.52083333333333
    zeros_result = 1.0
    ones_result = np.nan


class TestSleepPowerLawDistribution(_BaseTestEndpoint):
    metric = SleepPowerLawDistribution
    normal_result = 3.151675
    zeros_result = 1.0
    ones_result = 1 + 1 / np.log(30 / 29.5)


class TestWakePowerLawDistribution(_BaseTestEndpoint):
    metric = WakePowerLawDistribution
    normal_result = 2.073754
    zeros_result = 1 + 1 / np.log(30 / 29.5)
    ones_result = 1.0
