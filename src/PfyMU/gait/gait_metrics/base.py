"""
Base gait metrics class

Lukas Adamowicz
2020, Pfizer DMTI
"""
from numpy import zeros, diff, bool_


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

    @staticmethod
    def _get_mask1(gait_dict):
        mask = zeros(gait_dict['IC'].size, dtype=bool_)
        mask[:-1] = diff(gait_dict['Bout N']) == 0

        return mask

    @staticmethod
    def _get_mask2(gait_dict):
        mask = zeros(gait_dict['IC'].size, dtype=bool_)
        mask[:-2] = (gait_dict['Bout N'][2:] - gait_dict['Bout N'][:-2]) == 0

        return mask
