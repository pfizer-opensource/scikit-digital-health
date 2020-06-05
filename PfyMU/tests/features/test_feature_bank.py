from numpy import allclose

from PfyMU.features import *


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

        pred = feat_bank.compute(acc, 20.0)

        # check that values match truth values (ie doing correct computations)
        assert allclose(pred, bank_2d_truth)
        # check that the bank uses the same instance of the Range computation
        assert feat_bank._feat_list[1].parent is feat_bank._feat_list[3].parent  # check same instance

