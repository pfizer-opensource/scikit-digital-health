"""
Gait event-level and bout-level endpoint definitions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.


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
from numpy import (
    zeros,
    full,
    nanmean,
    nanstd,
    nanmedian,
    unique,
    sum,
    sqrt,
    nan,
    nonzero,
    argmin,
    abs,
    round,
    float_,
    int_,
    fft,
    arange,
    isnan,
    maximum,
    moveaxis,
    ascontiguousarray,
)
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt, find_peaks


from skdh.gait.gait_endpoints.base import (
    GaitEventEndpoint,
    GaitBoutEndpoint,
    basic_asymmetry,
)
from skdh.features.lib.extensions.statistics import autocorrelation
from skdh.features.lib.extensions.smoothness import SPARC


__all__ = [
    "StrideTime",
    "StanceTime",
    "SwingTime",
    "StepTime",
    "InitialDoubleSupport",
    "TerminalDoubleSupport",
    "DoubleSupport",
    "SingleSupport",
    "StepLength",
    "StrideLength",
    "GaitSpeed",
    "Cadence",
    "GaitSymmetryIndex",
    "IntraStepCovarianceV",
    "IntraStrideCovarianceV",
    "HarmonicRatioV",
    "StrideSPARC",
    "PhaseCoordinationIndex",
    "StepRegularityV",
    "StrideRegularityV",
    "AutocovarianceSymmetryV",
    "RegularityIndexV",
]


def _autocovariancefn(x, max_lag, biased=False, axis=0):
    y = ascontiguousarray(moveaxis(x, axis, -1))

    shape = list(y.shape)
    shape[-1] = max_lag
    ac = zeros(shape, dtype=float_)

    for i in range(min(max_lag, y.shape[-1] - 10)):
        ac[..., i] = autocorrelation(y, i, True)

    # if biased make sure its divided by just y.shape, instead of
    # y.shape - lag
    if biased:
        ac *= (y.shape[-1] - arange(0, max_lag)) / y.shape[-1]

    return moveaxis(ac, axis, -1)


# ===========================================================
#     GAIT EVENT-LEVEL ENDPOINTS
# ===========================================================
class StrideTime(GaitEventEndpoint):
    """
    The time to complete 1 full gait cycle for 1 foot. Defined as heel-strike (initial contact) to
    heel-strike for the same foot. A basic asymmetry measure is also computed as the difference
    between sequential stride times of opposite feet.
    """

    def __init__(self):
        super().__init__("stride time", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 2)
        gait[self.k_][mask] = (gait["IC"][mask_ofst] - gait["IC"][mask]) / fs


class StanceTime(GaitEventEndpoint):
    """
    The time during a stride in which the foot is on the ground. Defined as heel-strike
    (initial contact) to toe-off (final contact) for a foot. A basic asymmetry measure is also
    computed as the difference between sequential stance times of opposite feet.
    """

    def __init__(self):
        super().__init__("stance time", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        gait[self.k_] = (gait["FC"] - gait["IC"]) / fs


class SwingTime(GaitEventEndpoint):
    """
    The time during which the foot is off the ground. Defined as toe-off (final contact) to
    heel-strike (initial contact) of the same foot. A basic asymmetry measure is also computed as
    the difference between sequential swing times of opposite feet.
    """

    def __init__(self):
        super().__init__("swing time", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 2)
        gait[self.k_][mask] = (gait["IC"][mask_ofst] - gait["FC"][mask]) / fs


class StepTime(GaitEventEndpoint):
    """
    The duration from heel-strike (initial contact) to heel-strike of the opposite foot. A basic
    asymmetry measure is also computed as the difference between sequential step times of opposite
    feet.
    """

    def __init__(self):
        super().__init__("step time", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait["IC"][mask_ofst] - gait["IC"][mask]) / fs


class InitialDoubleSupport(GaitEventEndpoint):
    """
    The time immediately following heel strike during which the opposite foot is still on the
    ground. Defined as heel-strike (initial contact) to toe-off (final contact) of the opposite
    foot. A basic asymmetry measure is also computed as the difference between sequential initial
    double support times of opposite feet.
    """

    def __init__(self):
        super().__init__("initial double support", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        gait[self.k_] = (gait["FC opp foot"] - gait["IC"]) / fs


class TerminalDoubleSupport(GaitEventEndpoint):
    """
    The time immediately before toe-off (final contact) in which the opposite foot has contacted
    the ground. Defined as heel-strike (initial contact) of the opposite foot to toe-off of the
    current foot. A basic asymmetry measure is also computed as the difference between sequential
    terminal double support times of opposite feet.
    """

    def __init__(self):
        super().__init__("terminal double support", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        gait[self.k_][mask] = (
            gait["FC opp foot"][mask_ofst] - gait["IC"][mask_ofst]
        ) / fs


class DoubleSupport(GaitEventEndpoint):
    """
    The combined initial and terminal double support times. It is the total time during a stride
    that the current and opposite foot are in contact with the ground. A basic asymmetry measure
    is also computed as the difference between sequential double support times of opposite feet.
    """

    def __init__(self):
        super().__init__(
            "double support",
            __name__,
            depends=[InitialDoubleSupport, TerminalDoubleSupport],
        )

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        gait[self.k_] = (
            gait["PARAM:initial double support"] + gait["PARAM:terminal double support"]
        )


class SingleSupport(GaitEventEndpoint):
    """
    The time during a stride that only the current foot is in contact with the ground. Defined as
    opposite foot toe-off (final contact) to opposite foot heel-strike (initial contact). A basic
    asymmetry measure is also computed as the difference between sequential single support times
    of opposite feet.
    """

    def __init__(self):
        super().__init__("single support", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        gait[self.k_][mask] = (gait["IC"][mask_ofst] - gait["FC opp foot"][mask]) / fs


class StepLength(GaitEventEndpoint):
    """
    The distance traveled during a step (heel-strike to opposite foot heel-strike). A basic
    asymmetry measure is also computed as the difference between sequential step lengths of
    opposite feet.

    Notes
    -----
    The step length is computed using the inverted pendulum model from [1]_ per

    .. math:: L_{step} = 2\\sqrt{2l_{leg}h-h^2}

    where :math:`L_{step}` is the step length, :math:`l_{leg}` is the leg length, and
    :math:`h` is the Center of Mass change in height during a step. Leg length can either be
    measured, or taken to be :math:`0.53height`.

    References
    ----------
    .. [1] W. Zijlstra and A. L. Hof, “Assessment of spatio-temporal gait parameters from
        trunk accelerations during human walking,” Gait & Posture, vol. 18, no. 2, pp. 1–10,
        Oct. 2003, doi: 10.1016/S0966-6362(02)00190-X.
    """

    def __init__(self):
        super().__init__("step length", __name__)

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        if leg_length is not None:
            gait[self.k_] = 2 * sqrt(
                2 * leg_length * gait["delta h"] - gait["delta h"] ** 2
            )
        else:
            self._predict_init(gait, True, None)  # don't generate masks


class StrideLength(GaitEventEndpoint):
    r"""
    The distance traveled during a stride (heel-strike to current foot heel-strike). A basic
    asymmetry measure is also computed as the difference between sequential stride lengths of
    opposite feet.

    Notes
    -----
    The stride length is computed using the inverted pendulum model from [1]_ per

    .. math:: L_{step} = 2\sqrt{2l_{leg}h-h^2}
    .. math:: L_{stride} = L_{step, i} + L_{step, i+1}

    where :math:`L_{s}` is the step or stride length, :math:`l_{leg}` is the leg length, and
    :math:`h` is the Center of Mass change in height during a step. Leg length can either be
    measured, or taken to be :math:`0.53height`.

    References
    ----------
    .. [1] W. Zijlstra and A. L. Hof, “Assessment of spatio-temporal gait parameters from
        trunk accelerations during human walking,” Gait & Posture, vol. 18, no. 2, pp. 1–10,
        Oct. 2003, doi: 10.1016/S0966-6362(02)00190-X.
    """

    def __init__(self):
        super().__init__("stride length", __name__, depends=[StepLength])

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)
        if leg_length is not None:
            gait[self.k_][mask] = (
                gait["PARAM:step length"][mask_ofst] + gait["PARAM:step length"][mask]
            )


class GaitSpeed(GaitEventEndpoint):
    """
    How fast distance is being traveled. Defined as the stride length divided by the
    stride duration, in m/s. A basic asymmetry measure is also computed as the difference between
    sequential gait speeds of opposite feet.
    """

    def __init__(self):
        super().__init__("gait speed", __name__, depends=[StrideLength, StrideTime])

    @basic_asymmetry
    def _predict(self, fs, leg_length, gait, gait_aux):
        if leg_length is not None:
            gait[self.k_] = gait["PARAM:stride length"] / gait["PARAM:stride time"]
        else:
            self._predict_init(gait, True, None)  # don't generate masks


class Cadence(GaitEventEndpoint):
    """
    The number of steps taken in 1 minute. Computed per step as 60.0s divided by the step time.
    """

    def __init__(self):
        super().__init__("cadence", __name__, depends=[StepTime])

    def _predict(self, fs, leg_length, gait, gait_aux):
        gait[self.k_] = 60.0 / gait["PARAM:step time"]


class IntraStrideCovarianceV(GaitEventEndpoint):
    """
    The autocovariance of vertical acceleration of 1 stride with lag equal to the stride duration.
    In other words, it is how similar the vertical acceleration signal is from one stride to the
    next, for only 1 stride. It differs from the `StrideRegularity` in that stride regularity uses
    the acceleration from the entire gait bout while intra-stride covariance uses the acceleration
    only from individual steps. Values close to 1 indicate that the following stride was very
    similar to the current stride, while values close to 0 indicate that the following stride was
    not very similar.

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    def __init__(self):
        super().__init__("intra-stride covariance - V", __name__)

    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 2)

        i1 = gait["IC"][mask]
        i2 = gait["IC"][mask_ofst]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(mask)[0]):
            j_ = gait_aux["inertial data i"][idx]
            x = gait_aux["accel"][j_][i1[i] : i3[i], gait_aux["vert axis"][idx]]

            if (i3[i] - i1[i]) > x.size or x.size == 0:
                gait[self.k_][idx] = nan
            else:
                gait[self.k_][idx] = autocorrelation(x, i2[i] - i1[i], True)


class IntraStepCovarianceV(GaitEventEndpoint):
    """
    The autocovariance of vertical acceleration of 1 step with lag equal to the step duration. In
    other words, it is how similar the acceleration signal is from one step to the next for only
    1 step. It differs from the `StepRegularity` in that step regularity uses the acceleration
    from the entire gait bout while intra-step covariance uses the acceleration only from
    individual steps. Values close to 1 indicate that the following step was very
    similar to the current step, while values close to 0 indicate that the following step was
    not very similar.

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    def __init__(self):
        super().__init__("intra-step covariance - V", __name__)

    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, 1)

        i1 = gait["IC"][mask]
        i2 = gait["IC"][mask_ofst]
        i3 = i2 + (i2 - i1)

        for i, idx in enumerate(nonzero(mask)[0]):
            j_ = gait_aux["inertial data i"][idx]
            x = gait_aux["accel"][j_][i1[i] : i3[i], gait_aux["vert axis"][idx]]

            if (i3[i] - i1[i]) > x.size or x.size == 0:
                gait[self.k_][idx] = nan
            else:
                gait[self.k_][idx] = autocorrelation(x, i2[i] - i1[i], True)


class HarmonicRatioV(GaitEventEndpoint):
    r"""
    Symmetry measure of the 2 steps that occur during each stride. It attempts to capture this
    relationship by looking at the frequency components for steps and strides and creating a ratio
    using these values. Higher values indicate better symmetry between the steps occuring during
    an individual stride.

    Notes
    -----
    The Harmonic ratio is computed from the first 20 harmonics extracted from a fourier series.
    For the vertical direction, the HR is defined as

    .. math:: HR = \frac{\sum_{n=1}^{10}F(2nf_{stride})}{\sum_{n=1}^{10}F((2n-1)f_{stride})}

    where :math:`F` is the power spectral density and :math:`f_{stride}` is the stride frequency.
    Since this is computed on a per-stride basis, the stride frequency is estimated as the inverse
    of stride time for the individual stride.

    References
    ----------
    .. [1] J. L. Roche, K. A. Lowry, J. M. Vanswearingen, J. S. Brach, and M. S. Redfern,
        “Harmonic Ratios: A quantification of step to step symmetry,” J Biomech, vol. 46, no. 4,
        pp. 828–831, Feb. 2013, doi: 10.1016/j.jbiomech.2012.12.008.
    .. [2] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    def __init__(self):
        super().__init__("harmonic ratio - V", __name__, depends=[StrideTime])
        self._freq = fft.rfftfreq(
            1024
        )  # precompute the frequencies (still need to be scaled)
        # TODO add check for stride frequency, if too low, bump this up higher?
        self._harmonics = arange(1, 21, dtype=int_)

    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, init=True, offset=2)

        i1 = gait["IC"][mask]
        i2 = gait["IC"][mask_ofst]

        for i, idx in enumerate(nonzero(mask)[0]):
            va = gait_aux["vert axis"][idx]  # shorthand
            F = abs(
                fft.rfft(
                    gait_aux["accel"][gait_aux["inertial data i"][idx]][
                        i1[i] : i2[i], va
                    ],
                    n=1024,
                )
            )
            stridef = 1 / gait["PARAM:stride time"][idx]  # current stride frequency
            # get the indices for the first 20 harmonics
            ix_stridef = argmin(abs(self._freq * fs - stridef)) * self._harmonics
            if (ix_stridef < F.size).sum() <= 10:
                self.logger.warning(
                    f"High stride frequency [{stridef:.2f}] results too few harmonics in "
                    f"frequency range. Setting to nan"
                )
                gait[self.k_][idx] = nan
                continue
            elif (ix_stridef < F.size).sum() < 20:
                self.logger.warning(
                    f"High stride frequency [{stridef:.2f}] results in use of less than 20 "
                    f"harmonics [{(ix_stridef < F.size).sum()}]."
                )
            ix_stridef = ix_stridef[
                ix_stridef < F.size
            ]  # make sure not taking more than possible

            # index 1 is harmonic 2 -> even harmonics / odd harmonics
            gait[self.k_][idx] = sum(F[ix_stridef[1::2]]) / sum(F[ix_stridef[::2]])


class StrideSPARC(GaitEventEndpoint):
    r"""
    Assessment of the smoothness of the acceleration signal during a stride. SPARC is the
    spectral arc length, which is a measure of how smooth a signal is. Higher values (smaller
    negative) numbers indicate smoother strides.

    Notes
    -----
    Per the recommendation from [1]_, the SPARC is computed on the magnitude of the acceleration
    signal less gravity, during each stride.

    References
    ----------
    .. [1] S. Balasubramanian, A. Melendez-Calderon, A. Roby-Brami, and E. Burdet, “On the
        analysis of movement smoothness,” J NeuroEngineering Rehabil, vol. 12, no. 1, p. 112,
        Dec. 2015, doi: 10.1186/s12984-015-0090-9.
    """

    def __init__(self):
        super().__init__("stride SPARC", __name__)

    def _predict(self, fs, leg_length, gait, gait_aux):
        mask, mask_ofst = self._predict_init(gait, True, offset=2)

        i1 = gait["IC"][mask]
        i2 = gait["IC"][mask_ofst]

        for i, idx in enumerate(nonzero(mask)[0]):
            bout_i = gait_aux["inertial data i"][idx]

            if i2[i] - i1[i] > 0:
                gait[self.k_][idx] = SPARC(
                    norm(gait_aux["accel"][bout_i][i1[i] : i2[i], :], axis=1) - 1,
                    fs,  # fsample
                    4,  # padlevel
                    10.0,  # fcut
                    0.05,  # amplitude threshold
                )
            else:
                gait[self.k_][idx] = nan


# ===========================================================
#     GAIT BOUT-LEVEL ENDPOINTS
# ===========================================================
class PhaseCoordinationIndex(GaitBoutEndpoint):
    r"""
    Assessment of the symmetry between steps during straight overground gait.
    Computed for an entire bout, it is a measure of the deviation from symmetrical steps (ie half a
    stride is equal to exactly 1 step duration). Lower values indicate better symmetry and
    a "more consistent and accurate phase generation" [2]_.

    Notes
    -----
    The computation of PCI relies on the assumption that healthy gait is perfectly even, with
    step times being exactly half of stride times. This assumption informs the definition
    of the PCI, where the perfect step phase is set to :math:`180^\circ`. To compute PCI, the
    phase is first computed for each stride as the relative step to stride time in degrees,

    .. math:: \varphi_i = 360^\circ\left(\frac{hs_{i+1}-hs_{i}}{hs_{i+2}-hs{i}}\right)

    where :math:`hs_i` is the *ith* heel-strike. Then over the whole bout, the mean absolute
    difference from :math:`180^\circ` is computed as :math:`\varphi_{ABS}`,

    .. math:: \varphi_{ABS} = \frac{1}{N}\sum_{i=1}^{N}|\varphi_i - 180^\circ|

    The coefficient of variation (:math:`\varphi_{CV}`) is also computed for phase,

    .. math: \varphi_{CV} = 100\frac{s_{\varphi}}{\bar{\varphi}}

    where :math:`\bar{\varphi}` and :math:`s_{\varphi}` are the sample mean and standard deviation
    of :math:`\varphi` respectively. Finally, the PCI is computed per

    .. math:: PCI = \varphi_{CV} + 100\frac{\varphi_{ABS}}{180}

    References
    ----------
    .. [1] M. Plotnik, N. Giladi, and J. M. Hausdorff, “A new measure for quantifying the
        bilateral coordination of human gait: effects of aging and Parkinson’s disease,”
        Exp Brain Res, vol. 181, no. 4, pp. 561–570, Aug. 2007, doi: 10.1007/s00221-007-0955-7.
    .. [2] A. Weiss, T. Herman, N. Giladi, and J. M. Hausdorff, “Association between Community
        Ambulation Walking Patterns and Cognitive Function in Patients with Parkinson’s Disease:
        Further Insights into Motor-Cognitive Links,” Parkinsons Dis, vol. 2015, 2015,
        doi: 10.1155/2015/547065.
    """

    def __init__(self):
        super().__init__(
            "phase coordination index", __name__, depends=[StrideTime, StepTime]
        )

    def _predict(self, fs, leg_length, gait, gait_aux):
        pci = zeros(len(gait_aux["accel"]), dtype=float_)

        phase = gait["PARAM:step time"] / gait["PARAM:stride time"]  # %, not degrees
        for i in range(len(gait_aux["accel"])):
            mask = gait_aux["inertial data i"] == i

            psi_abs = nanmean(abs(phase[mask] - 0.5))  # using % not degrees right now
            psi_cv = nanstd(phase[mask], ddof=1) / nanmean(phase[mask])

            pci[i] = 100 * (psi_cv + psi_abs / 0.5)

        gait[self.k_] = pci[gait_aux["inertial data i"]]


class GaitSymmetryIndex(GaitBoutEndpoint):
    r"""
    Assessment of the symmetry between steps during straight overground gait. It is computed for
    an entire bout. Values closer to 1 indicate higher symmetry, while values close to 0 indicate
    lower symmetry.

    Notes
    -----
    If the minimum gait window time is less than 4.5 seconds, there may be issues with this
    endpoint for those with slow gait (those with stride lengths approaching the minimum gait
    window time).

    GSI is computed using the biased autocovariance of the acceleration after being filtered
    through a 4th order 10Hz cutoff butterworth low-pass filter. [1]_ and [2]_ use the
    autocorrelation, instead of autocovariance, however subtracting from the compared signals
    results in a better mathematical comparison of the symmetry of the acceleration profile of the
    gait. The biased autocovariance is used to suppress the value at higher lags [1]_. In order to
    ensure that full steps/strides are capture, the maximum lag for the autocorrelation is set to
    4s, which should include several strides in healthy adults, and account for more than
    2.5 strides in impaired populations, such as hemiplegic stroke patients [3]_.

    With the autocovariances computed for all 3 acceleration axes, the coefficient of stride
    repetition (:math:`C_{stride}`) is computed for lag :math:`m` per

    .. math:: C_{stride}(m) = K_{AP}(m) + K_{V}(m) + K_{ML}(m)

    where :math:`K_{x}` is the autocovariance in the :math:`x` direction - Anterior-Posterior (AP),
    Medial-Lateral (ML), or vertical (V). The coefficient of step repetition (:math:`C_{step}`)
    is the norm of :math:`C_{stride}`

    .. math:: C_{step}(m) = \sqrt{C_{stride}(m)} = \sqrt{K_{AP}(m) + K_{V}(m) + K_{ML}(m)}

    Under the assumption that perfectly symmetrical gait will have step durations equal to half
    the stride duration, the GSI is computed per

    .. math:: GSI = C_{step}(0.5m_{stride}) / \sqrt{3}

    where :math:`m_{stride}` is the lag for the average stride in the gait bout, and corresponds to
    a local maximum in the autocovariance function. To find the peak corresponding to
    :math:`m_{stride}` the peak nearest to the average stride time for the bout is used. GSI is
    normalized by :math:`\sqrt{3}` in order to have a maximum value of 1.

    References
    ----------
    .. [1] W. Zhang, M. Smuck, C. Legault, M. A. Ith, A. Muaremi, and K. Aminian, “Gait Symmetry
        Assessment with a Low Back 3D Accelerometer in Post-Stroke Patients,” Sensors, vol. 18,
        no. 10, p. 3322, Oct. 2018, doi: 10.3390/s18103322.
    .. [2] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    .. [3] H. P. von Schroeder, R. D. Coutts, P. D. Lyden, E. Billings, and V. L. Nickel, “Gait
        parameters following stroke: a practical assessment,” Journal of Rehabilitation Research
        and Development, vol. 32, no. 1, pp. 25–31, Feb. 1995.
    """

    def __init__(self):
        super().__init__("gait symmetry index", __name__, depends=[StrideTime])

    def _predict(self, fs, leg_length, gait, gait_aux):
        gsi = zeros(len(gait_aux["accel"]), dtype=float_)

        # setup acceleration filter if its possible to use
        if 0 < (2 * 10 / fs) < 1:
            sos = butter(4, 2 * 10 / fs, btype="low", output="sos")
        else:
            sos = None

        for i, acc in enumerate(gait_aux["accel"]):
            lag_ = (
                nanmedian(gait["PARAM:stride time"][gait_aux["inertial data i"] == i])
                * fs
            )
            if isnan(lag_):  # if only nan values in the bout
                gsi[i] = nan
                continue
            lag = int(round(lag_))
            # GSI uses biased autocovariance
            if sos is not None:
                ac = _autocovariancefn(
                    sosfiltfilt(sos, acc, axis=0), int(4.5 * fs), biased=True, axis=0
                )
            else:
                ac = _autocovariancefn(acc, int(4.5 * fs), biased=True, axis=0)

            # C_stride is the sum of 3 axes
            pks, _ = find_peaks(sum(ac, axis=1))
            # find the closest peak to the computed ideal half stride lag
            try:
                t_stride = pks[argmin(abs(pks - lag))]
                idx = int(0.5 * t_stride)

                # maximum ensures no sqrt of negative numbers
                gsi[i] = sqrt(sum(maximum(ac[idx], 0))) / sqrt(3)
            except ValueError:
                gsi[i] = nan

        gait[self.k_] = gsi[gait_aux["inertial data i"]]


class StepRegularityV(GaitBoutEndpoint):
    """
    The autocovariance at a lag time of 1 step for the vertical acceleration. Computed for an
    entire bout of gait, it is a measure of the average symmetry of sequential steps during
    overground strait gait. Values close to 1 indicate high degree of regularity/symmetry, while
    values close to 0 indicate a low degree of regularity/symmetry.

    Notes
    -----
    If the minimum gait window time is less than 4.5 seconds, there may be issues with this
    endpoint for those with slow gait (those with stride lengths approaching the minimum gait
    window time).

    Step regularity is the value of the autocovariance function at a lag equal to the time
    for one step. While [2]_ uses the autocorrelation instead of the autocovariance like [1]_, the
    autocovariance is used here as it provides a mathematically better comparison of the
    acceleration profile during gait.

    The peak corresponding to one step time is found by searching the area near the lag
    corresponding to the average step time for the gait bout. The nearest peak to this point is
    used as the peak at a lag of one step.

    References
    ----------
    .. [1] R. Moe-Nilssen and J. L. Helbostad, “Estimation of gait cycle characteristics by trunk
        accelerometry,” Journal of Biomechanics, vol. 37, no. 1, pp. 121–126, Jan. 2004,
        doi: 10.1016/S0021-9290(03)00233-1.
    .. [2] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    def __init__(self):
        super().__init__("step regularity - V", __name__, depends=[StepTime])

    def _predict(self, fs, leg_length, gait, gait_aux):
        stepreg = full(len(gait_aux["accel"]), nan, dtype=float_)

        for i in unique(gait_aux["inertial data i"]):
            acc = gait_aux["accel"][i]
            mask = gait_aux["inertial data i"] == i

            va = gait_aux["vert axis"][mask][0]
            lag_ = nanmedian(gait["PARAM:step time"][mask]) * fs
            if isnan(lag_):  # if only nan values in the bout
                stepreg[i] = nan
                continue
            lag = int(round(lag_))
            acf = _autocovariancefn(acc[:, va], int(4.5 * fs), biased=False, axis=0)
            pks, _ = find_peaks(acf)
            try:
                idx = pks[argmin(abs(pks - lag))]
                stepreg[i] = acf[idx]
            except ValueError:
                stepreg[i] = nan

        # broadcast step regularity into gait for each step
        gait[self.k_] = stepreg[gait_aux["inertial data i"]]


class StrideRegularityV(GaitBoutEndpoint):
    """
    Autocovariance at a lag time of 1 stride for the vertical acceleration. Computed for an
    entire bout of gait, it is a measure of the average symmetry of sequential stride during
    overground strait gait. Values close to 1 indicate high degree of regularity/symmetry, while
    values close to 0 indicate a low degree of regularity/symmetry.

    Notes
    -----
    If the minimum gait window time is less than 4.5 seconds, there may be issues with this
    endpoint for those with slow gait (those with stride lengths approaching the minimum gait
    window time).

    Stride regularity is the value of the autocovariance function at a lag equal to the time
    for one stride. While [2]_ uses the autocorrelation instead of the autocovariance like [1]_,
    the autocovariance is used here as it provides a mathematically better comparison of the
    acceleration profile during gait.

    The peak corresponding to one stride time is found by searching the area near the lag
    corresponding to the average stride time for the gait bout. The nearest peak to this point is
    used as the peak at a lag of one stride.

    References
    ----------
    .. [1] R. Moe-Nilssen and J. L. Helbostad, “Estimation of gait cycle characteristics by trunk
        accelerometry,” Journal of Biomechanics, vol. 37, no. 1, pp. 121–126, Jan. 2004,
        doi: 10.1016/S0021-9290(03)00233-1.
    .. [2] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    def __init__(self):
        super().__init__("stride regularity - V", __name__, depends=[StrideTime])

    def _predict(self, fs, leg_length, gait, gait_aux):
        stridereg = full(len(gait_aux["accel"]), nan, dtype=float_)

        for i in unique(gait_aux["inertial data i"]):
            acc = gait_aux["accel"][i]
            mask = gait_aux["inertial data i"] == i

            va = gait_aux["vert axis"][mask][0]
            lag_ = nanmedian(gait["PARAM:stride time"][mask]) * fs
            if isnan(lag_):  # if only nan values in the bout
                stridereg[i] = nan
                continue
            lag = int(round(lag_))
            acf = _autocovariancefn(acc[:, va], int(4.5 * fs), biased=False, axis=0)
            pks, _ = find_peaks(acf)
            try:
                idx = pks[argmin(abs(pks - lag))]
                stridereg[i] = acf[idx]
            except ValueError:
                stridereg[i] = nan

        # broadcast step regularity into gait for each step
        gait[self.k_] = stridereg[gait_aux["inertial data i"]]


class AutocovarianceSymmetryV(GaitBoutEndpoint):
    """
    The absolute difference between stride and step regularity for the vertical axis.
    It quantifies the level of symmetry between the stride and step regularity and provide an
    overall endpoint of symmetry for the gait bout

    Notes
    -----
    If the minimum gait window time is less than 4.5 seconds, there may be issues with this
    endpoint for those with slow gait (those with stride lengths approaching the minimum gait
    window time).

    References
    ----------
    .. [1] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    def __init__(self):
        super().__init__(
            "autocovariance symmetry - V",
            __name__,
            depends=[StepRegularityV, StrideRegularityV],
        )

    def _predict(self, fs, leg_length, gait, gait_aux):
        gait[self.k_] = abs(
            gait["BOUTPARAM:step regularity - V"]
            - gait["BOUTPARAM:stride regularity - V"]
        )


class RegularityIndexV(GaitBoutEndpoint):
    r"""
    The combination of both step and stride regularity into one endpoint. The goal is to provide an
    assessment of the regularity for consecutive steps and strides, for the vertical axis
    acceleration. Values closer to 1 indicate high levels of symmetry between left and right steps.

    Notes
    -----
    The vertical axis regularity index :math:`R_V` is simply defined per

    .. math:: R_V = 1 - |R_{(stride, V)} - R_{(step, V)}|\frac{2}{R_{(stride, V)} + R_{(step, V)}}

    where :math:`R_{(stride, V)}` is the stride regularity for the vertical axis (same notation for
    step regularity).

    The Regularity Index term came from [1]_, where it was defined without the subtraction from 1.
    However, the definition from [2]_ (under the term "symmetry") keeps the values in the same
    range as others (including step/stride regularity), aiding in ease of interpretation.
    "Regularity Index" however serves to eliminate confusion given other endpoints already labeled
    with "symmetry" in the name.

    References
    ----------
    .. [1] L. Angelini et al., “Is a Wearable Sensor-Based Characterisation of Gait Robust Enough
        to Overcome Differences Between Measurement Protocols? A Multi-Centric Pragmatic Study in
        Patients with Multiple Sclerosis,” Sensors, vol. 20, no. 1, Art. no. 1, Jan. 2020,
        doi: 10.3390/s20010079.
    .. [2] L. Angelini et al., “Wearable sensors can reliably quantify gait alterations associated
        with disability in people with progressive multiple sclerosis in a clinical setting,”
        J Neurol, vol. 267, no. 10, pp. 2897–2909, Oct. 2020, doi: 10.1007/s00415-020-09928-8.


    """

    def __init__(self):
        super().__init__(
            "regularity index - V",
            __name__,
            depends=[StepRegularityV, StrideRegularityV],
        )

    def _predict(self, fs, leg_length, gait, gait_aux):
        str_v = "BOUTPARAM:stride regularity - V"
        ste_v = "BOUTPARAM:step regularity - V"

        gait[self.k_] = 1 - (
            2 * abs(gait[str_v] - gait[ste_v]) / (gait[ste_v] + gait[str_v])
        )
