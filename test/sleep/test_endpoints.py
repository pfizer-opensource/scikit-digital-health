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
        sleep_pred = np.zeros(40)
        l, s, v = rle(sleep_pred)

        res = self.metric().predict(
            sleep_predictions=sleep_pred,
            lengths=l,
            starts=s,
            values=v
        )

        assert np.isclose(res, self.zeros_result, equal_nan=True)

    def test_all_ones(self):
        sleep_pred = np.ones(40)
        l, s, v = rle(sleep_pred)

        res = self.metric().predict(
            sleep_predictions=sleep_pred,
            lengths=l,
            starts=s,
            values=v
        )

        assert np.isclose(res, self.ones_result, equal_nan=True)


class TestSleepAverageHazard(_BaseTestEndpoint):
    metric = SleepAverageHazard
    normal_result = 0.8333333
    zeros_result = np.nan
    ones_result = 1.0
