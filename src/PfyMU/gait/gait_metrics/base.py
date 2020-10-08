"""
Base gait metrics class

Lukas Adamowicz
2020, Pfizer DMTI
"""
import functools

from numpy import zeros, roll, full, nan, bool_, float_


def basic_asymmetry(f):
    @functools.wraps(f)
    def run_basic_asymmetry(self, *args, **kwargs):
        f(self, *args, **kwargs)
        self._predict_asymmetry(*args, **kwargs)
    return run_basic_asymmetry


class EventMetric:
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self, name, depends=None):
        """
        Gait metric base class

        Parameters
        ----------
        name : str
            Name of the metric
        """
        self.name = name
        self.k_ = f'PARAM:{self.name}'

        self._depends = depends

    @staticmethod
    def _get_mask(gait, offset):
        if offset not in [1, 2]:
            raise ValueError('invalid offset')
        mask = zeros(gait['IC'].size, dtype=bool_)
        mask[:-offset] = (gait['Bout N'][offset:] - gait['Bout N'][:-offset]) == 0

        return mask

    def predict(self, dt, leg_length, gait, gait_aux):
        """
        Predict the gait metric

        Parameters
        ----------
        dt : float
            Sampling period in seconds
        leg_length : {None, float}
            Leg length in meters
        gait : dict
            Dictionary of gait items and results. Modified in place to add the metric being
            calculated
        gait_aux : dict
            Dictionary of acceleration, velocity, and position data for bouts, and the mapping
            from step to bout and inertial data
        """
        if self.k_ in gait:
            return
        if self._depends is not None:
            for param in self._depends:
                param().predict(dt, leg_length, gait, gait_aux)

        self._predict(dt, leg_length, gait, gait_aux)

    def _predict_asymmetry(self, dt, leg_length, gait, gait_aux):
        asy_name = f'{self.name} asymmetry'
        gait[asy_name] = full(gait['IC'].size, nan, dtype=float_)

        mask = self._get_mask(gait, 1)
        mask_ofst = roll(mask, 1)

        gait[asy_name][mask] = gait[self.k_][mask_ofst] - gait[self.k_][mask]

    def _predict_init(self, gait, init=True, offset=None):
        if init:
            gait[self.k_] = full(gait['IC'].size, nan, dtype=float_)
        if offset is not None:
            mask = self._get_mask(gait, offset)
            mask_ofst = roll(mask, offset)
            return mask, mask_ofst
