"""
Gait metric definitions

Lukas Adamowicz
2020, Pfizer DMTI


     IC             FC      IC             FC      IC             FC
     i-1            i-1     i+1            i+1     i+3            i+3
L    |--------------|       |--------------|       |--------------|
R               |--------------|        |--------------|
                i              i        i+2            i+2
                IC             FC       IC             FC

For step/stride starting at IC_i
stride: IC_i+2 - IC_i
stance: FC_i   - IC_i
swing:  IC_i+2 - FC_i   // IC_i+2 - FCof_i+1 :: stride - stance = (IC_i+2 - IC_i) - FC_i + IC_i
step:   IC_i+1 - IC_i
ids:    FC_i-1 - IC_i   // FCof_i - IC_i
tds:    FC_i   - IC_i+1 // FCof_i+1 - IC_i+1
tds:    ids + tds
ss:     IC_i+1 - FC_i-1 // IC_i+1 - FCof_i
ss:     stance - tds = - FC_i-1 + IC_i+1[ + IC_i - IC_i - FC_i + FC_i]

h = signal_range(vpos_IC_i : vpos_IC_i+1)
step length: 2 * sqrt(2 * l * h - h**2)
stride length: step_length_i + step_length_i+1

gait speed: stride_length / stride time
"""
from numpy import std, cov, sqrt, nan, nonzero, abs


from PfyMU.gait.gait_metrics.base import GaitMetric, basic_asymmetry


__all__ = ['StrideTime', 'StanceTime', 'SwingTime', 'StepTime', 'InitialDoubleSupport',
           'TerminalDoubleSupport', 'DoubleSupport', 'SingleSupport', 'StepLength',
           'StrideLength', 'GaitSpeed', 'Cadence', 'StepRegularity', 'StrideRegularity']


def _autocov(x, i1, i2, i3):
    if i3 > x.size:
        return nan
    else:
        ac = cov(x[i1:i2], x[i2:i3], bias=False)[0, 1]
        return ac / (std(x[i1:i2], ddof=1) * std(x[i2:i3], ddof=1))


class StrideTime(GaitMetric):
    def __init__(self):
        super().__init__('stride time')

    @basic_asymmetry
    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 2)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['IC'][mask]) * dt


class StanceTime(GaitMetric):
    def __init__(self):
        super().__init__('stance time')

    def predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = (gait['FC'] - gait['IC']) * dt


class SwingTime(GaitMetric):
    def __init__(self):
        super().__init__('swing time')
    
    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 2)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['FC'][mask]) * dt


class StepTime(GaitMetric):
    def __init__(self):
        super().__init__('step time')
    
    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['IC'][mask]) * dt


class InitialDoubleSupport(GaitMetric):
    def __init__(self):
        super().__init__('initial double support')

    def predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = (gait['FC opp foot'] - gait['IC']) * dt


class TerminalDoubleSupport(GaitMetric):
    def __init__(self):
        super().__init__('terminal double support')
    
    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait['FC opp foot'][mask_ofst] - gait['IC'][mask_ofst]) * dt


class DoubleSupport(GaitMetric):
    def __init__(self):
        super().__init__('double support', depends=[InitialDoubleSupport, TerminalDoubleSupport])

    def predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = gait['PARAM:initial double support'] + gait['PARAM:terminal double support']


class SingleSupport(GaitMetric):
    def __init__(self):
        super().__init__('single support')
    
    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['FC opp foot'][mask]) * dt


class StepLength(GaitMetric):
    def __init__(self):
        super().__init__('step length')
    
    def predict(self, dt, leg_length, gait, gait_aux):
        if leg_length is not None:
            gait[self.k_] = 2 * sqrt(2 * leg_length * gait['delta h'] - gait['delta h']**2)


class StrideLength(GaitMetric):
    def __init__(self):
        super().__init__('stride length', depends=[StepLength])

    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 1)
        if leg_length is not None:
            gait[self.k_] = gait['PARAM:step length'][mask_ofst] \
                            + gait['PARAM:step length'][mask]


class GaitSpeed(GaitMetric):
    def __init__(self):
        super().__init__('gait speed', depends=[StrideLength, StrideTime])

    def predict(self, dt, leg_length, gait, gait_aux):
        if leg_length is not None:
            gait[self.k_] = gait['PARAM:stride length'] / gait['PARAM:stride time']


class Cadence(GaitMetric):
    def __init__(self):
        super().__init__('cadence', depends=[StepTime])

    def predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = 60.0 / gait['PARAM:step time']


class StrideRegularity(GaitMetric):
    def __init__(self):
        super().__init__('stride regularity - V')

    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 2)

        i1 = gait['IC'][mask]
        i2 = gait['IC'][mask_ofst]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(mask)[0]):
            gait[self.k_][idx] = _autocov(
                gait_aux['vert accel'][gait_aux['inertial data i'][idx]],
                i1[i], i2[i], i3[i]
            )


class StepRegularity(GaitMetric):
    def __init__(self):
        super().__init__('step regularity - V')

    def predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self.__predict_init(gait, True, 1)

        i1 = gait['IC'][mask]
        i2 = gait['IC'][mask_ofst]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(mask)[0]):
            gait[self.k_][idx] = _autocov(
                gait_aux['vert accel'][gait_aux['inertial data i'][idx]],
                i1[i], i2[i], i3[i]
            )


class AutocorrelationSymmetry(GaitMetric):
    def __init__(self):
        super().__init__(
            'autocorrelation symmetry - V', depends=[StepRegularity, StrideRegularity]
        )

    def predict(self, dt, leg_length, gait, gait_aux):
        super().predict(dt, leg_length, gait, gait_aux)

        gait[self.k_] = abs(
            gait['PARAM:step regularity - V'] - gait['PARAM:stride regularity - V']
        )
