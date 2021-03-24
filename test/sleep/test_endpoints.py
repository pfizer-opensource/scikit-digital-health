import numpy as np

from skimu.sleep.utility import rle
from skimu.sleep.endpoints import *


class _BaseTestEndpoint:
    def test(self, dummy_sleep_predictions):
        l, s, v = rle(dummy_sleep_predictions)

        res = self.metric().predict(
            sleep_predictions=dummy_sleep_predictions,
            lengths=l,
            starts=s,
            values=v
        )

        assert np.isclose(res, self.normal_result, equal_nan=True)

    def test_all_zeros(self):
        sleep_pred = np.zeros(30)
        l, s, v = rle(sleep_pred)

        res = self.metric().predict(
            sleep_predictions=sleep_pred,
            lengths=l,
            starts=s,
            values=v
        )

        assert np.isclose(res, self.zeros_result, equal_nan=True)

    def test_all_ones(self):
        sleep_pred = np.ones(30)
        l, s, v = rle(sleep_pred)

        res = self.metric().predict(
            sleep_predictions=sleep_pred,
            lengths=l,
            starts=s,
            values=v
        )

        assert np.isclose(res, self.ones_result, equal_nan=True)


class TestTotalSleepTime(_BaseTestEndpoint):
    metric = TotalSleepTime
    normal_result = 13
    zeros_result = 0
    ones_result = 30


class TestPercentTimeAsleep(_BaseTestEndpoint):
    metric = PercentTimeAsleep
    normal_result = 13/30 * 100
    zeros_result = 0.
    ones_result = 100.


class TestNumberWakeBouts(_BaseTestEndpoint):
    metric = NumberWakeBouts
    normal_result = 2
    zeros_result = 0
    ones_result = 0


class TestSleepOnsetLatency(_BaseTestEndpoint):
    metric = SleepOnsetLatency
    normal_result = 3
    zeros_result = np.nan
    ones_result = 0.


class TestWakeAfterSleepOnset(_BaseTestEndpoint):
    metric = WakeAfterSleepOnset
    normal_result = 7
    zeros_result = np.nan
    ones_result = 0.

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
