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
from numpy import std, cov, full, nan, roll, nonzero, float_


from PfyMU.gait.gait_metrics.base import GaitMetric


class StepRegularity(GaitMetric):
    def __init__(self):
        super().__init__('step regularity - V')

    def __call__(self, gait_dict, gait_aux):
        """
        Compute the parameter

        Parameters
        ----------
        gait_dict : dictionary
        gait_aux : dictionary

        Returns
        -------
        step_regularity
        """
        # get the masks for +1, +2
        m1 = self._get_mask1(gait_dict)
        m1_1 = roll(m1, 1)

        gait_dict[f'PARAM:{self.name}'] = full(gait_dict['IC'].size, nan, dtype=float_)

        i1 = gait_dict['IC'][m1]
        i2 = gait_dict['IC'][m1_1]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(m1)[0]):
            gait_dict[f'PARAM:{self.name}'][idx] = self._autocov(
                gait_aux['vert accel'][gait_aux['inertial data i']],
                i1[i], i2[i], i3[i]
            )

    @staticmethod
    def _autocov(x, i1, i2, i3):
        if i3 > x.size:
            return nan
        else:
            ac = cov(x[i1:i2], x[i2:i3], bias=False)[0, 1]
            return ac / (std(x[i1:i2], ddof=1) * std(x[i2:i3], ddof=1))
