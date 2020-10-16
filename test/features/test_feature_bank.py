import pytest
from numpy import allclose, random
import pandas as pd

from skimu.features import *
from skimu.features.core import NoFeaturesError, NotAFeatureError


class TestBank:
    def set_up(self):
        bank = Bank(window_length=None, window_step=None)

        bank + Mean()
        bank + Range()[['x', 'z']]
        bank + JerkMetric(normalize=True)
        bank + Range()['y']

        return bank

    def test_2d_ndarray(self, fs, acc, bank_2d_truth):
        feat_bank = self.set_up()

        _ = feat_bank.compute(acc, 20.0, columns=['x', 'y', 'z'])
        pred = feat_bank.compute(acc, 20.0)  # run 2x to catch any bugs with multiple runs on same data

        # check that values match truth values (ie doing correct computations)
        assert allclose(pred, bank_2d_truth)
        # check that the bank uses the same instance of the Range computation
        assert feat_bank._feat_list[1].parent is feat_bank._feat_list[3].parent  # check same instance

    def test_dataframe(self, fs, acc, bank_2d_truth):
        feat_bank = self.set_up()

        acc_df = pd.DataFrame(data=acc, columns=['x', 'y', 'z'])

        _ = feat_bank.compute(acc_df, 20.0)
        pred = feat_bank.compute(acc_df, 20.0)  # run 2x to catch any bugs with multiple runs on same data

        # check that values match truth values (ie doing correct computations)
        assert allclose(pred.values, bank_2d_truth)
        # check that the bank uses the same instance of the Range computation
        assert feat_bank._feat_list[1].parent is feat_bank._feat_list[3].parent  # check same instance

    def test_no_features_error(self):
        feat_bank = Bank(window_length=None, window_step=None)

        with pytest.raises(NoFeaturesError):
            feat_bank.compute(None, None, None, None)

    def test_columns_size_error(self):
        feat_bank = self.set_up()

        with pytest.raises(ValueError):
            feat_bank.compute(random.rand(150, 10, 3), 50.0, ['x', 'y'], False)

