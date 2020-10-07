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
from numpy import mean, std, sum, sqrt, nan, nonzero, abs
from scipy.signal import butter, sosfiltfilt


from PfyMU.gait.gait_metrics.base import GaitMetric, basic_asymmetry


__all__ = ['StrideTime', 'StanceTime', 'SwingTime', 'StepTime', 'InitialDoubleSupport',
           'TerminalDoubleSupport', 'DoubleSupport', 'SingleSupport', 'StepLength',
           'StrideLength', 'GaitSpeed', 'Cadence', 'GaitSymmetryIndex', 'StepRegularity',
           'StrideRegularity', 'AutocorrelationSymmetry']


def _autocovariance(x, i1, i2, i3, biased=False):
    if i3 > x.size:
        return nan

    N = i3 - i1
    m = i2 - i1
    m1, s1 = mean(x[i1:i2]), std(x[i1:i2], ddof=1)
    m2, s2 = mean(x[i2:i3]), std(x[i2:i3], ddof=1)

    ac = sum((x[i1:i2] - m1) * (x[i2:i3] - m2))
    if biased:
        ac /= (N * s1 * s2)
    else:
        ac /= ((N - m) * s1 * s2)

    return ac


def _autocovariance3(x, i1, i2, i3, biased=False):
    if i3 > x.shape[0]:
        return nan

    N = i3 - i1
    m = i2 - i1
    m1, s1 = mean(x[i1:i2], axis=0, keepdims=True), std(x[i1:i2], ddof=1, axis=0, keepdims=True)
    m2, s2 = mean(x[i2:i3], axis=0, keepdims=True), std(x[i2:i3], ddof=1, axis=0, keepdims=True)

    ac = sum((x[i1:i2] - m1) * (x[i2:i3] - m2), axis=0)
    if biased:
        ac /= (N * s1 * s2)
    else:
        ac /= ((N - m) * s1 * s2)

    return ac


class StrideTime(GaitMetric):
    """
    Stride time is the time to complete 1 full gait cycle for 1 foot. Defined as heel-strike
    (initial contact) to heel-strike for the same foot.

    A basic asymmetry measure is also computed as the difference between sequential
    stride times of opposite feet
    """
    def __init__(self):
        super().__init__('stride time')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 2)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['IC'][mask]) * dt


class StanceTime(GaitMetric):
    """
    Stance time is the time during which the foot is on the ground. Defined as heel-strike
    (initial contact) to toe-off (final contact) for a foot.

    A basic asymmetry measure is also computed as the difference between sequential
    stance times of opposite feet
    """
    def __init__(self):
        super().__init__('stance time')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = (gait['FC'] - gait['IC']) * dt


class SwingTime(GaitMetric):
    """
    Swing time is the time during which the foot is off the ground. Defined as toe-off
    (final contact) to heel-strike (initial contact) of the same foot.

    A basic asymmetry measure is also computed as the difference between sequential
    swing times of opposite feet
    """
    def __init__(self):
        super().__init__('swing time')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 2)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['FC'][mask]) * dt


class StepTime(GaitMetric):
    """
    Step time is the duration from heel-strike (initial contact) to heel-strike of the opposite
    foot.

    A basic asymmetry measure is also computed as the difference between sequential
    step times of opposite feet
    """
    def __init__(self):
        super().__init__('step time')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['IC'][mask]) * dt


class InitialDoubleSupport(GaitMetric):
    """
    Initial double support is the time immediately following heel strike during which the
    opposite foot is still on the ground. Defined as heel-strike (initial contact) to toe-off
    (final contact) of the opposite foot.

    A basic asymmetry measure is also computed as the difference between sequential
    initial double support times of opposite feet
    """
    def __init__(self):
        super().__init__('initial double support')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = (gait['FC opp foot'] - gait['IC']) * dt


class TerminalDoubleSupport(GaitMetric):
    """
    Terminal double support is the time immediately before toe-off (final contact) in which
    the opposite foot has contacted the ground. Defined as heel-strike (initial contact) of the
    opposite foot to toe-off of the current foot

    A basic asymmetry measure is also computed as the difference between sequential
    terminal double support times of opposite feet
    """
    def __init__(self):
        super().__init__('terminal double support')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait['FC opp foot'][mask_ofst] - gait['IC'][mask_ofst]) * dt


class DoubleSupport(GaitMetric):
    """
    Double support is the combined initial and terminal double support times. It is the total
    time during a stride that the current and opposite foot are in contact with the ground.

    A basic asymmetry measure is also computed as the difference between sequential
    double support times of opposite feet
    """
    def __init__(self):
        super().__init__('double support', depends=[InitialDoubleSupport, TerminalDoubleSupport])

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = gait['PARAM:initial double support'] \
                        + gait['PARAM:terminal double support']


class SingleSupport(GaitMetric):
    """
    Single support is the time during a stride that only the current foot is in contact with
    the ground. Defined as opposite foot toe-off (final contact) to opposite foot heel-strike
    (initial contact).

    A basic asymmetry measure is also computed as the difference between sequential
    single support times of opposite feet
    """
    def __init__(self):
        super().__init__('single support')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait['IC'][mask_ofst] - gait['FC opp foot'][mask]) * dt


class StepLength(GaitMetric):
    """
    Step length is the distance traveled during a step (heel-strike to opposite foot
    heel-strike). Here it is computed using the inverted pendulum model from [1]_:

    :math:`L_{step} = 2\\sqrt{2l_{leg}h-h^2}`

    where :math:`L_{step}` is the step length, :math:`l_{leg}` is the leg length, and
    :math:`h` is the Center of Mass change in height during a step. Leg length can either be
    measured, or taken to be :math:`0.53height`

    A basic asymmetry measure is also computed as the difference between sequential
    step lengths of opposite feet

    References
    ----------
    .. [1] W. Zijlstra and A. L. Hof, “Assessment of spatio-temporal gait parameters from
        trunk accelerations during human walking,” Gait & Posture, vol. 18, no. 2, pp. 1–10,
        Oct. 2003, doi: 10.1016/S0966-6362(02)00190-X.
    """
    def __init__(self):
        super().__init__('step length')

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        if leg_length is not None:
            gait[self.k_] = 2 * sqrt(2 * leg_length * gait['delta h'] - gait['delta h']**2)
        else:
            self._predict_init(gait, True, None)  # don't generate masks


class StrideLength(GaitMetric):
    """
    Stride length is the distance traveled during a stride (heel-strike to current foot
    heel-strike). Here it is computed using the inverted pendulum model from [1]_:

    :math:`L_{step} = 2\\sqrt{2l_{leg}h-h^2}`
    :math:`L_{stride} = L_{step, i} + L_{step, i+1}`

    where :math:`L_{s}` is the step or stride length, :math:`l_{leg}` is the leg length, and
    :math:`h` is the Center of Mass change in height during a step. Leg length can either be
    measured, or taken to be :math:`0.53height`

    A basic asymmetry measure is also computed as the difference between sequential
    stride lengths of opposite feet

    References
    ----------
    .. [1] W. Zijlstra and A. L. Hof, “Assessment of spatio-temporal gait parameters from
        trunk accelerations during human walking,” Gait & Posture, vol. 18, no. 2, pp. 1–10,
        Oct. 2003, doi: 10.1016/S0966-6362(02)00190-X.
    """
    def __init__(self):
        super().__init__('stride length', depends=[StepLength])

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        if leg_length is not None:
            gait[self.k_][mask] = gait['PARAM:step length'][mask_ofst] \
                            + gait['PARAM:step length'][mask]


class GaitSpeed(GaitMetric):
    """
    Gait speed is how fast distance is being convered. Defined as the stride length divided by the
    stride duration, in m/s

    A basic asymmetry measure is also computed as the difference between sequential
    gait speeds of opposite feet
    """
    def __init__(self):
        super().__init__('gait speed', depends=[StrideLength, StrideTime])

    @basic_asymmetry
    def _predict(self, dt, leg_length, gait, gait_aux):
        if leg_length is not None:
            gait[self.k_] = gait['PARAM:stride length'] / gait['PARAM:stride time']
        else:
            self._predict_init(gait, True, None)  # don't generate masks


class Cadence(GaitMetric):
    """
    Cadence is the number of steps taken in 1 minute. Here it is computed per step, as 60.0s
    divided by the step time
    """
    def __init__(self):
        super().__init__('cadence', depends=[StepTime])

    def _predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = 60.0 / gait['PARAM:step time']


class GaitSymmetryIndex(GaitMetric):
    """
    Gait Symmetry Index (GSI) assesses symmetry between steps during straight overground gait.

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """
    def __init__(self):
        super().__init__('gait symmetry index')

    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)

        i1 = gait['IC'][mask]
        i2 = gait['IC'][mask_ofst]
        i3 = i2 + (i2 - i1)

        # filter the acceleration data first
        faccel = {'accel': gait_aux['accel']}
        sos = butter(4, 2 * 10 * dt, btype='low', output='sos')
        for i, acc in enumerate(faccel['accel']):
            faccel['accel'][i] = sosfiltfilt(sos, acc, axis=0)

        for i, idx in enumerate(nonzero(mask)[0]):
            ac = _autocovariance3(
                faccel['accel'][gait_aux['inertial data i'][idx]],
                i1[i], i2[i], i3[i]
            )
            gait[self.k_][idx] = sqrt(sum(ac)) / sqrt(3)


class StrideRegularity(GaitMetric):
    """
    Stride regularity is the autocovariance with lag equal to 1 stride duration. In other words,
    it is how similar the acceleration signal is from one stride to the next. Values near 1
    indicate very symmetrical strides

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    .. [2] R. Moe-Nilssen and J. L. Helbostad, “Estimation of gait cycle characteristics by trunk
        accelerometry,” Journal of Biomechanics, vol. 37, no. 1, pp. 121–126, Jan. 2004,
        doi: 10.1016/S0021-9290(03)00233-1.

    """
    def __init__(self):
        super().__init__('stride regularity - V')

    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 2)

        i1 = gait['IC'][mask]
        i2 = gait['IC'][mask_ofst]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(mask)[0]):
            gait[self.k_][idx] = _autocovariance(
                # index the accel, then the list of views, then the vertical axis
                gait_aux['accel'][gait_aux['inertial data i'][idx]][:, gait_aux['vert axis']],
                i1[i], i2[i], i3[i], biased=False
            )


class StepRegularity(GaitMetric):
    """
    Step regularity is the autocovariance with lag equal to 1 step duration.  In other words, it
    is how similar the acceleration signal is from one step to the next. Values near 1 indicate
    very symmetrical steps

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    .. [2] R. Moe-Nilssen and J. L. Helbostad, “Estimation of gait cycle characteristics by trunk
        accelerometry,” Journal of Biomechanics, vol. 37, no. 1, pp. 121–126, Jan. 2004,
        doi: 10.1016/S0021-9290(03)00233-1.
    """
    def __init__(self):
        super().__init__('step regularity - V')

    def _predict(self, dt, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)

        i1 = gait['IC'][mask]
        i2 = gait['IC'][mask_ofst]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(mask)[0]):
            gait[self.k_][idx] = _autocovariance(
                gait_aux['accel'][gait_aux['inertial data i'][idx]][:, gait_aux['vert axis']],
                i1[i], i2[i], i3[i], biased=False
            )


class AutocorrelationSymmetry(GaitMetric):
    """
    Autocorrelation symmetry is the absolute difference between stride and step regularity.

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """
    def __init__(self):
        super().__init__(
            'autocorrelation symmetry - V', depends=[StepRegularity, StrideRegularity]
        )

    def _predict(self, dt, leg_length, gait, gait_aux):
        gait[self.k_] = abs(
            gait['PARAM:step regularity - V'] - gait['PARAM:stride regularity - V']
        )
