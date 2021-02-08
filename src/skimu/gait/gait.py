"""
Gait detection, processing, and analysis from wearable inertial sensor data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from collections.abc import Iterable
from warnings import warn

import h5py
from numpy import mean, diff, arange, zeros, interp, float_, abs, asarray, sum, argmin, ndarray

from skimu.base import _BaseProcess
from skimu.gait.get_gait_classification import get_gait_classification_lgbm, DimensionMismatchError
from skimu.gait.get_gait_bouts import get_gait_bouts
from skimu.gait.get_gait_events import get_gait_events
from skimu.gait.get_strides import get_strides
from skimu.gait.gait_metrics import gait_metrics
from skimu.gait.gait_metrics import EventMetric, BoutMetric


class LowFrequencyError(Exception):
    pass


def get_downsampled_data(time, accel, gait_pred, fs, goal_fs, days, downsample):
    """
    Get the downsampled data from input data

    Parameters
    ----------
    time
    accel
    gait_pred
    goal_fs
    downsample
    kw

    Returns
    -------
    time_ds
    accel_ds
    gait_pred_ds
    days
    """
    if downsample:
        _days = asarray(days)
        time_ds = arange(time[0], time[-1], 1 / goal_fs)
        accel_ds = zeros((time_ds.size, 3), dtype=float_)
        for i in range(3):
            accel_ds[:, i] = interp(time_ds, time, accel[:, i])

        if isinstance(gait_pred, ndarray):
            if gait_pred.size != accel.shape[0]:
                raise DimensionMismatchError(
                    "Number of gait predictions must match number of acceleration samples")
            gait_pred_ds = interp(time_ds, time, gait_pred)
        else:
            gait_pred_ds = gait_pred

        days = zeros(asarray(_days).shape)
        for i, day_idx in enumerate(_days):
            i_guess = (day_idx * goal_fs / fs).astype(int) - 1000
            i_guess[i_guess < 0] = 0
            days[i, 0] = argmin(abs(time_ds[i_guess[0]:i_guess[0] + 2000] - time[day_idx[0]]))
            days[i, 1] = argmin(abs(time_ds[i_guess[1]:i_guess[1] + 2000] - time[day_idx[1]]))
            days[i] += i_guess  # make sure to add the starting index back in

        return time_ds, accel_ds, gait_pred_ds, days
    else:
        return time, accel, gait_pred, days


class Gait(_BaseProcess):
    """
    Process IMU data to extract metrics of gait. Detect gait, extract gait events (heel-strikes,
    toe-offs), and compute gait metrics from inertial data collected from a lumbar mounted
    wearable inertial measurement unit

    Parameters
    ----------
    correct_accel_orient : bool, optional
        Correct the acceleration orientation using the method from [7]_. This should only be '
        applied if the accelerometer axes are already approximately aligned with the anatomical
        axes. The correction is applied on a per-gait-bout basis. Default is True.
    use_cwt_scale_relation : bool, optional
        Use the optimal scale/frequency relationship (see Notes). This changes which
        scale is used for the smoothing/differentiation operation performed with the
        continuous wavelet transform. Default is True. See Notes for a caveat of the
        relationship
    min_bout_time : float, optional
        Minimum time in seconds for a gait bout. Default is 8s (making a minimum of 3 3-second
        windows)
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
    prov_leg_length : bool, optional
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
    .. [7] R. Moe-Nilssen, “A new method for evaluating motor control in gait under real-life
        environmental conditions. Part 1: The instrument,” Clinical Biomechanics, vol. 13, no.
        4–5, pp. 320–327, Jun. 1998, doi: 10.1016/S0268-0033(98)00089-8.
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
        gait_metrics.StrideSPARC,
        # bout level metrics
        gait_metrics.PhaseCoordinationIndex,
        gait_metrics.GaitSymmetryIndex,
        gait_metrics.StepRegularityV,
        gait_metrics.StrideRegularityV,
        gait_metrics.AutocovarianceSymmetryV,
        gait_metrics.RegularityIndexV
    ]

    def __init__(
            self,
            correct_accel_orient=True,
            use_cwt_scale_relation=True,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            max_stride_time=2.25,
            loading_factor=0.2,
            height_factor=0.53,
            prov_leg_length=False,
            filter_order=4,
            filter_cutoff=20.0
    ):
        super().__init__(
            # key-word arguments for storage
            correct_accel_orient=correct_accel_orient,
            use_cwt_scale_relation=use_cwt_scale_relation,
            min_bout_time=min_bout_time,
            max_bout_separation_time=max_bout_separation_time,
            max_stride_time=max_stride_time,
            loading_factor=loading_factor,
            height_factor=height_factor,
            prov_leg_length=prov_leg_length,
            filter_order=filter_order,
            filter_cutoff=filter_cutoff
        )

        self.corr_accel_orient = correct_accel_orient
        self.use_opt_scale = use_cwt_scale_relation
        self.min_bout = min_bout_time
        self.max_bout_sep = max_bout_separation_time

        self.max_stride_time = max_stride_time
        self.loading_factor = loading_factor

        if prov_leg_length:
            self.height_factor = 1.0
        else:
            self.height_factor = height_factor

        self.filt_ord = filter_order
        self.filt_cut = filter_cutoff

        # for saving gait predictions
        self._save_classifier_fn = lambda time, starts, stops: None

    def _save_classifier_predictions(self, fname):
        def fn(time, starts, stops):
            with h5py.File(fname, 'w') as f:
                f['time'] = time
                f['bout starts'] = starts
                f['bout stops'] = stops
        self._save_classifier_fn = fn

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
                raise ValueError("Not all objects are EventMetric or BoutMetric")
        else:
            if isinstance(metrics(), (EventMetric, BoutMetric)):
                self._params.append(metrics)
            else:
                raise ValueError(f'Metric {metrics!r} is not a EventMetric or BoutMetric')

    def predict(self, time=None, accel=None, *, gyro=None, height=None, gait_pred=None, **kwargs):
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
        super().predict(
            time=time, accel=accel, gyro=gyro, height=height, gait_pred=gait_pred, **kwargs
        )

        if height is None:
            warn("height not provided, not computing spatial metrics", UserWarning)
            leg_length = None
        else:
            # height factor is set to 1 if providing leg length
            leg_length = self.height_factor * height

        # compute fs/delta t
        fs = 1 / mean(diff(time))
        if fs < 20.0:
            raise LowFrequencyError(f"Frequency ({fs:.2f}Hz) is too low (<20Hz).")
        if fs < (50.0 * 0.985):  # 1.5% margin
            warn("Frequency is less than 50Hz. Downsampling to 20Hz. Note that this may effect "
                 "gait metric results values", UserWarning)

        # downsample acceleration
        goal_fs = 50 if fs > (50 * 0.985) else 20
        downsample = False if ((0.985 * goal_fs) < fs < (1.015 * goal_fs)) else True

        days = kwargs.get(self._days, [[0, accel.shape[0] - 1]])
        time_ds, accel_ds, gait_pred_ds, days = get_downsampled_data(
            time, accel, gait_pred, fs, goal_fs, days, downsample)

        # original scale. Compute outside loop since stays the same
        # 1.25 comes from original paper, corresponds to desired frequency
        # 0.2 is the central frequency of the 'gaus1' wavelet (normalized to 1)
        original_scale = max(round(0.2 / (1.25 / goal_fs)), 1)

        # setup the storage for the gait parameters
        gait = {
            i: [] for i in [
                'Day N', 'Bout N', 'Bout Starts', 'Bout Duration', 'Bout Steps', 'Gait Cycles',
                'IC', 'FC', 'FC opp foot', 'valid cycle', 'delta h'
            ]
        }
        # aux dictionary for storing values for computing gait metrics
        gait_aux = {
            i: [] for i in
            ['vert axis', 'accel', 'vert velocity', 'vert position', 'inertial data i']
        }

        # get the gait classification if necessary
        gbout_starts, gbout_stops = get_gait_classification_lgbm(gait_pred_ds, accel_ds, goal_fs)
        self._save_classifier_fn(time_ds, gbout_starts, gbout_stops)

        gait_i = 0  # keep track of where everything is in the loops

        for iday, day_idx in enumerate(days):
            start, stop = day_idx

            # GET GAIT BOUTS
            # ==============
            gait_bouts = get_gait_bouts(
                gbout_starts, gbout_stops, start, stop, time_ds, self.max_bout_sep, self.min_bout
            )

            for ibout, bout in enumerate(gait_bouts):
                # get the gait events, vertical acceleration, and vertical axis
                ic, fc, vert_acc, v_axis = get_gait_events(
                    accel_ds[bout],
                    goal_fs,
                    time_ds[bout],
                    original_scale,
                    self.filt_ord,
                    self.filt_cut,
                    self.corr_accel_orient,
                    self.use_opt_scale
                )

                # get the strides
                strides_in_bout = get_strides(
                    gait, vert_acc, gait_i, ic, fc, time_ds[bout], goal_fs, self.max_stride_time,
                    self.loading_factor
                )

                # add inertial data to the aux dict for use in gait metric calculation
                gait_aux['accel'].append(accel_ds[bout, :])
                # add the index for the corresponding accel/velocity/position
                gait_aux['inertial data i'].extend([len(gait_aux['accel']) - 1] * strides_in_bout)
                gait_aux['vert axis'].extend([v_axis] * strides_in_bout)

                # save some default per bout metrics
                gait['Bout N'].extend([ibout + 1] * strides_in_bout)
                gait['Bout Starts'].extend([time_ds[bout.start]] * strides_in_bout)
                gait['Bout Duration'].extend(
                    [(bout.stop - bout.start) / goal_fs] * strides_in_bout
                )

                gait['Bout Steps'].extend([strides_in_bout] * strides_in_bout)
                gait['Gait Cycles'].extend([sum(gait['valid cycle'][gait_i:])] * strides_in_bout)

                gait_i += strides_in_bout

            # add the day number
            gait['Day N'].extend([iday + 1] * (len(gait['Bout N']) - len(gait['Day N'])))

        # convert to arrays
        for key in gait:
            gait[key] = asarray(gait[key])
        # convert inertial data index to an array
        gait_aux['inertial data i'] = asarray(gait_aux['inertial data i'])
        gait_aux['vert axis'] = asarray(gait_aux['vert axis'])

        # loop over metrics and compute
        for param in self._params:
            param().predict(goal_fs, leg_length, gait, gait_aux)

        # remove invalid gait cycles
        for k in [i for i in gait if i != 'valid cycle']:
            gait[k] = gait[k][gait['valid cycle']]

        # remove unnecessary stuff from gait dict
        gait.pop('IC', None)
        gait.pop('FC', None)
        gait.pop('FC opp foot', None)
        gait.pop('valid cycle', None)

        kwargs.update({self._acc: accel, self._time: time, self._gyro: gyro, 'height': height})
        if self._in_pipeline:
            return kwargs, gait
        else:
            return gait
