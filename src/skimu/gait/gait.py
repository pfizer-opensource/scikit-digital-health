"""
Gait detection, processing, and analysis from wearable inertial sensor data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn
from collections.abc import Iterable

from numpy import ndarray, mean, diff, abs, argmax, sign, round, array, full, bool_

from skimu.base import _BaseProcess
from skimu.gait.get_gait_classification import get_gait_classification_lgbm
from skimu.gait.get_gait_bouts import get_gait_bouts
from skimu.gait.get_gait_events import get_gait_events
from skimu.gait.get_strides import get_strides
from skimu.gait import gait_metrics
from skimu.gait.gait_metrics import EventMetric, BoutMetric
from skimu.gait.get_bout_metrics_delta_h import get_bout_metrics_delta_h


class LowFrequencyError(Exception):
    pass


class Gait(_BaseProcess):
    """
    Process IMU data to extract metrics of gait. Detect gait, extract gait events (heel-strikes,
    toe-offs), and compute gait metrics from inertial data collected from a lumbar mounted
    wearable inertial measurement unit

    Parameters
    ----------
    use_cwt_scale_relation : bool, optional
        Use the optimal scale/frequency relationship (see Notes). This changes which
        scale is used for the smoothing/differentiation operation performed with the
        continuous wavelet transform. Default is True. See Notes for a caveat of the
        relationship
    min_bout_time : float, optional
        Minimum time in seconds for a gait bout. Default is 5s
    max_bout_separation_time : float, optional
        Maximum time in seconds between two bouts of gait for them to be merged into 1 gait
        bout. Default is 0.5s
    max_stride_time : float, optional
        The maximum time in seconds for a stride, for optimization of gait events detection.
        Default is 2.25s
    loading_factor : float, optional
        The factor to determine maximum loading time (initial double support time), for
        optimization of gait events detection. Default is 0.2
    height_factor : float, optional
        The factor multiplied by height to obtain an estimate of leg length.
        Default is 0.53 [4]_. Ignored if `leg_length` is `True`
    leg_length : bool, optional
        If the actual leg length will be provided. Setting to true would have the same effect
        as setting height_factor to 1.0 while providing leg length. Default is False
    filter_order : int, optional
        Acceleration low-pass filter order. Default is 4
    filter_cutoff : float, optional
        Acceleration low-pass filter cutoff in Hz. Default is 20.0Hz

    Notes
    -----
    The optimal scale/frequency relationship found in [5]_ was based on a cohort of only young
    women students. While it is recommended to use this relationship, the user should be aware
    of this shortfall in the generation of the relationship.

    3 optimizations are performed on the detected events to minimize false positives.

    1. Loading time (initial double support) must be less than
    :math:`loading\\_factor * max\\_stride\\_time`

    2. Stance time must be less than
    :math:`(max\\_stride\\_time/2) + loading\\_factor * max\\_stride\\_time`

    3. Stride time must be less than `max\\_stride\\_time`

    References
    ----------
    .. [1] B. Najafi, K. Aminian, A. Paraschiv-Ionescu, F. Loew, C. J. Bula, and P. Robert,
        “Ambulatory system for human motion analysis using a kinematic sensor: monitoring of
        daily physical activity in the elderly,” IEEE Transactions on Biomedical Engineering,
        vol. 50, no. 6, pp. 711–723, Jun. 2003, doi: 10.1109/TBME.2003.812189.
    .. [2] W. Zijlstra and A. L. Hof, “Assessment of spatio-temporal gait parameters from
        trunk accelerations during human walking,” Gait & Posture, vol. 18, no. 2, pp. 1–10,
        Oct. 2003, doi: 10.1016/S0966-6362(02)00190-X.
    .. [3] J. McCamley, M. Donati, E. Grimpampi, and C. Mazzà, “An enhanced estimate of initial
        contact and final contact instants of time using lower trunk inertial sensor data,”
        Gait & Posture, vol. 36, no. 2, pp. 316–318, Jun. 2012,
        doi: 10.1016/j.gaitpost.2012.02.019.
    .. [4] S. Del Din, A. Godfrey, and L. Rochester, “Validation of an Accelerometer to
        Quantify a Comprehensive Battery of Gait Characteristics in Healthy Older Adults and
        Parkinson’s Disease: Toward Clinical and at Home Use,” IEEE J. Biomed. Health Inform.,
        vol. 20, no. 3, pp. 838–847, May 2016, doi: 10.1109/JBHI.2015.2419317.
    .. [5] C. Caramia, C. De Marchis, and M. Schmid, “Optimizing the Scale of a Wavelet-Based
        Method for the Detection of Gait Events from a Waist-Mounted Accelerometer under
        Different Walking Speeds,” Sensors, vol. 19, no. 8, p. 1869, Jan. 2019,
        doi: 10.3390/s19081869.
    .. [6] C. Buckley et al., “Gait Asymmetry Post-Stroke: Determining Valid and Reliable
        Methods Using a Single Accelerometer Located on the Trunk,” Sensors, vol. 20, no. 1,
        Art. no. 1, Jan. 2020, doi: 10.3390/s20010037.
    """

    # gait parameters
    _params = [
        # event level metrics
        gait_metrics.StrideTime,
        gait_metrics.StanceTime,
        gait_metrics.SwingTime,
        gait_metrics.StepTime,
        gait_metrics.InitialDoubleSupport,
        gait_metrics.TerminalDoubleSupport,
        gait_metrics.DoubleSupport,
        gait_metrics.SingleSupport,
        gait_metrics.StepLength,
        gait_metrics.StrideLength,
        gait_metrics.GaitSpeed,
        gait_metrics.Cadence,
        gait_metrics.IntraStepCovarianceV,
        gait_metrics.IntraStrideCovarianceV,
        gait_metrics.HarmonicRatioV,
        # bout level metrics
        gait_metrics.PhaseCoordinationIndex,
        gait_metrics.GaitSymmetryIndex,
        gait_metrics.StepRegularityV,
        gait_metrics.StrideRegularityV,
        gait_metrics.AutocovarianceSymmetryV,
        gait_metrics.RegularityIndexV
    ]

    def __repr__(self):
        ret = "Gait("
        ret += f"use_cwt_scale_relation={self.use_opt_scale!r}, "
        ret += f"min_bout_time={self.min_bout!r}, "
        ret += f"max_bout_separation_time={self.max_bout_sep!r}, "
        ret += f"max_stride_time={self.max_stride_time!r}, "
        ret += f"loading_factor={self.loading_factor!r}, "
        ret += f"height_factor={self.height_factor!r}, "
        ret += f"leg_length={self.leg_length!r}, "
        ret += f"filter_order={self.filt_ord!r}, "
        ret += f"filter_cutoff={self.filt_cut!r})"
        return ret

    def __init__(self, use_cwt_scale_relation=True, min_bout_time=5.0,
                 max_bout_separation_time=0.5, max_stride_time=2.25, loading_factor=0.2,
                 height_factor=0.53, leg_length=False, filter_order=4, filter_cutoff=20.0):
        super().__init__('Gait Process', True)

        self.use_opt_scale = use_cwt_scale_relation
        self.min_bout = min_bout_time
        self.max_bout_sep = max_bout_separation_time

        self.max_stride_time = max_stride_time
        self.loading_factor = loading_factor

        self.height_factor = height_factor
        self.leg_length = leg_length

        self.filt_ord = filter_order
        self.filt_cut = filter_cutoff

    def add_metrics(self, metrics):
        """
        Add metrics to be computed

        Parameters
        ----------
        metrics : {Iterable, callable}
            Either an iterable of EventMetric or BoutMetric references or an individual
            EventMetric/BoutMetric reference to be added to the list of metrics to be computed

        Examples
        --------
        >>> class NewGaitMetric(gait_metrics.EventMetric):
        >>>     pass
        >>>
        >>> gait = Gait()
        >>> gait.add_metrics(NewGaitMetric)

        >>> class NewGaitMetric(gait_metrics.EventMetric):
        >>>     pass
        >>> class NewGaitMetric2(gait_metrics.BoutMetric):
        >>>     pass
        >>>
        >>> gait = Gait()
        >>> gait.add_metrics([NewGaitMetric, NewGaitMetric2])
        """
        if isinstance(metrics, Iterable):
            if all(isinstance(i(), (EventMetric, BoutMetric)) for i in metrics):
                self._params.extend(metrics)
            else:
                raise ValueError('Must provide either a GaitMetric or iterable of GaitMetrics')
        else:
            if isinstance(metrics(), gait_metrics.EventMetric):
                self._params.append(metrics)

    def predict(self, *args, **kwargs):
        """
        predict(time, accel, *, gyro=None, height=None, gait_pred=None)

        Get the gait events and metrics from a time series signal

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by centrally mounted lumbar device, in
            units of 'g'
        gyro : numpy.ndarray, optional
            (N, 3) array of angular velocities measured by the same centrally mounted lumbar
            device, in units of 'deg/s'. Only optionally used if provided. Main functionality
            is to allow distinguishing step sides.
        height : float, optional
            Either height (False) or leg length (True) of the subject who wore the inertial
            measurement device, in meters, depending on `leg_length`. If not provided,
            spatial metrics will not be computed
        gait_pred : {any, numpy.ndarray}, optional
            (N, ) array of boolean predictions of gait, or any value that is not None. If not an
            ndarray but not None, the entire recording will be taken as gait. If not provided
            (or None), gait classification will be performed on the acceleration data.

        Returns
        -------
        gait_results : dict
            The computed gait metrics. For a list of metrics and their definitions, see
            :ref:`event-level-gait-metrics` and :ref:`bout-level-gait-metrics`.

        Raises
        ------
        LowFrequencyError
            If the sampling frequency is less than 20Hz
        """
        return super().predict(*args, **kwargs)

    def _predict(self, time=None, accel=None, *, gyro=None, height=None, gait_pred=None,
                 **kwargs):
        """
        predict(time=None, accel=None, *, gyro=None, height=None, gait_pred=None)

        Get the gait events and metrics from a time series signal

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by centrally mounted lumbar device, in
            units of 'g'
        gyro : numpy.ndarray, optional
            (N, 3) array of angular velocities measured by the same centrally mounted lumbar
            device, in units of 'deg/s'. Only optionally used if provided. Main functionality
            is to allow distinguishing step sides.
        height : float, optional
            Either height (False) or leg length (True) of the subject who wore the inertial
            measurement device, in meters, depending on `leg_length`. If not provided,
            spatial metrics will not be computed
        gait_pred : numpy.ndarray, optional
            (N, ) array of boolean predictions of gait. If not provided, gait classification
            will be performed on the acceleration data

        Returns
        -------
        gait_results : dict
        """
        if height is None:
            warn('height not provided, not computing spatial metrics', UserWarning)
            leg_length = None
        else:
            # if not providing leg length (default), multiply height by the height factor
            if not self.leg_length:
                leg_length = self.height_factor * height
            else:
                leg_length = height

        # compute fs/delta t
        dt = mean(diff(time[:500]))

        if (1 / dt) < 20.0:
            raise LowFrequencyError(f"Frequency ({1/dt:.2f}Hz) is too low (<20Hz)")

        # check if windows exist for days
        if self._days in kwargs:
            days = kwargs[self._days]
        else:
            days = [(0, accel.shape[0])]

        # get the gait classifications if necessary
        if gait_pred is None:
            gait_pred = get_gait_classification_lgbm(accel, 1 / dt)
        else:
            if isinstance(gait_pred, ndarray):
                if gait_pred.size != accel.shape[0]:
                    raise ValueError(
                        'Number of gait predictions must much number of accel samples')
            else:
                gait_pred = full(accel.shape[0], True, dtype=bool_)

        # figure out vertical axis
        acc_mean = mean(accel, axis=0)
        v_axis = argmax(abs(acc_mean))

        # original scale. compute outside loop.
        # 1.25 comes from original paper, corresponds to desired frequency
        # 0.4 comes from using the 'gaus1' wavelet
        scale_original = round(0.4 / (2 * 1.25 * dt)) - 1

        gait = {
            'Day N': [],
            'Bout N': [], 'Bout Start': [], 'Bout Duration': [], 'Bout Steps': [],
            'IC': [], 'FC': [], 'FC opp foot': [],
            'b valid cycle': [], 'delta h': []
        }
        # auxiliary dictionary for storing values for computing gait metrics
        gait_aux = {
            'vert axis': v_axis,
            'accel': [],
            'vert velocity': [],
            'vert position': [],
            'inertial data i': []
        }

        ig = 0  # keep track of where everything is in the cycle

        for iday, day_idx in enumerate(days):
            start, stop = day_idx

            # GET GAIT BOUTS
            # ======================================
            gait_bouts = get_gait_bouts(
                gait_pred[start:stop], dt, self.max_bout_sep, self.min_bout
            )

            for ibout, bout in enumerate(gait_bouts):
                bstart = start + bout[0]

                ic, fc, vert_acc = get_gait_events(
                    accel[bstart:start + bout[1], v_axis],
                    dt,
                    sign(acc_mean[v_axis]),
                    scale_original,
                    self.filt_ord,
                    self.filt_cut,
                    self.use_opt_scale
                )

                # add inertial data to the aux dict for use in gait metric calculation
                gait_aux['accel'].append(accel[bstart:start+bout[1], :])

                # get strides
                sib = get_strides(gait, ig, ic, fc, dt, self.max_stride_time, self.loading_factor)

                # add the index for the corresponding accel/velocity/position
                gait_aux['inertial data i'].extend([len(gait_aux['accel']) - 1] * sib)

                # get the initial gait metrics
                get_bout_metrics_delta_h(
                    gait, ig, ibout, dt, time, vert_acc, sib, bout, bstart
                )

                ig += sib

            # add the day number
            gait['Day N'].extend([iday + 1] * (len(gait['Bout N']) - len(gait['Day N'])))

        # convert to arrays
        for key in gait:
            gait[key] = array(gait[key])
        # convert inertial data index to an array
        gait_aux['inertial data i'] = array(gait_aux['inertial data i'])

        # loop over metrics and compute
        for param in self._params:
            param().predict(dt, leg_length, gait, gait_aux)

        kwargs.update({self._acc: accel, self._time: time, self._gyro: gyro, 'height': height})
        return kwargs, gait
