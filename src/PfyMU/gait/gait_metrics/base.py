"""
Base gait metrics class

Lukas Adamowicz
2020, Pfizer DMTI
"""
from numpy import zeros, roll, full, nan, bool_, float_


class GaitMetric:
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self, name):
        """
        Gait metric base class

        Parameters
        ----------
        name : str
            Name of the metric
        """
        self.name = name

        self.k_ = f'PARAM:{self.name}'

    @staticmethod
    def _get_mask(gait, offset):
        if offset not in [1, 2]:
            raise ValueError('invalid offset')
        mask = zeros(gait['IC'].size, dtype=bool_)
        mask[:-offset] = (gait['Bout N'][offset:] - gait['Bout N'][:-offset]) == 0

        return mask

    def predict(self, dt, leg_length, gait, gait_aux):
        pass

    def __predict_init(self, gait, init=True, offset=None):
        if init:
            gait[self.k_] = full(gait['IC'].size, nan, dtype=float_)
        if offset is not None:
            mask = self._get_mask(gait, offset)
            mask_ofst = roll(mask, offset)
            return mask, mask_ofst
