"""
Gait detection, processing, and analysis from wearable inertial sensor data

Lukas Adamowicz
Copyright (c) 2023, Pfizer Inc. All rights reserved.
"""
from collections.abc import Iterable
from warnings import warn

from numpy import ndarray, asarray, mean, diff, sum, nan

from skdh import Pipeline
from skdh.base import BaseProcess, handle_process_returns
from skdh.utility.internal import rle, apply_downsample
from skdh.utility.exceptions import LowFrequencyError

from skdh.gaitv3.gait_endpoints import GaitEventEndpoint, GaitBoutEndpoint
from skdh.gaitv3.gait_endpoints import gait_endpoints
from skdh.gaitv3.utility import get_gait_bouts
from skdh.gaitv3 import substeps


class GaitLumbar(BaseProcess):
    """
    Process lumbar IMU data to extract metrics of gait. Detect gait, extract gait
    events (heel-strikes, toe-offs), and compute gait metrics from inertial data
    collected from a lumbar mounted wearable inertial measurement unit. If angular
    velocity data is provided, turns are detected, and steps during turns are noted.

    Parameters
    ----------
    downsample : bool, optional
        Downsample acceleration data to either 50hz (for sampling rates >50hz) or
        20hz (for sampling rates <50hz but >20hz). Default is False.
    height_factor : float, optional
        The factor multiplied by height to obtain an estimate of leg length.
        Default is 0.53 [4]_. Ignored if `leg_length` is `True`
    provide_leg_length : bool, optional
        If the actual leg length will be provided. Setting to true would have the same effect
        as setting height_factor to 1.0 while providing leg length. Default is False.
    min_bout_time : float, optional
        Minimum time in seconds for a gait bout. Default is 8s (making a minimum
        of 3 3-second windows).
    max_bout_separation_time : float, optional
        Maximum time in seconds between two bouts of gait for them to be merged into
        1 gait bout. Default is 0.5s.
    gait_event_method : {"AP CWT", "Vertical CWT"}, optional
        The method to use for gait event detection, case-insensitive. "AP CWT"
        uses :meth:`skdh.gaitv3.substeps.ApCwtGaitEvents`, while "Vertical CWT"
        uses :meth:`skdh.gaitv3.substeps.VerticalCwtGaitEvents`. Default is "AP CWT".
    correct_orientation : bool, optional
        Correct the accelerometer orientation if it is slightly mis-aligned
        with the anatomical axes. Default is True. Used in the pre-processing
        step of the bout processing pipeline.
    filter_cutoff : float, optional
        [:meth:`skdh.gaitv3.substeps.PreprocessGaitBout`] Low-pass filter cutoff
        in Hz. Default is 20.0.
    filter_order : int, optional
        [:meth:`skdh.gaitv3.substeps.PreprocessGaitBout`] Low-pass filter order.
        Default is 4.
    use_cwt_scale_relation : bool, optional
        [:meth:`skdh.gaitv3.substeps.VerticalCwtGaitEvents`] Use the optimal scale/frequency
        relationship.
    wavelet_scale : {"default", float, int}, optional
        [:meth:`skdh.gaitv3.substeps.VerticalCwtGaitEvents`] The wavelet scale to use.
    max_stride_time : {callable, float}, optional
        [:meth:`skdh.gaitv3.substeps.CreateStridesAndQc`] Definition of how the maximum
        stride time is calculated. Either a callable with the input of the mean step time,
        or a float, which will be used as a static limit. Default is the function
        `2.0 * mean_step_time + 1.0`.
    loading_factor : {callable, float}, optional
        [:meth:`skdh.gaitv3.substeps.CreateStridesAndQc`] Definition of the loading factor.
        Either a callable with the input of mean step time, or a float (between 0.0
        and 1.0) indicating a static factor. Default is the function
        `0.17 * mean_step_time + 0.05`.

    Other Parameters
    ----------------
    bout_processing_pipeline : {None, Pipeline}, optional
        The processing pipeline to use on bouts of gait. Default is None, which
        creates a standard pipeline (see Notes). If you need more than these
        steps, you can provide your own pipeline. NOTE that you must set
        `flatten_results=True` when creating the custom pipeline.

    Notes
    -----
    The default pipeline is the following steps:
    - :meth:`skdh.gaitv3.substeps.PreprocessGaitBout`
    - :meth:`skdh.gaitv3.substeps.ApCwtGaitEvents` or
    :meth:`skdh.gaitv3.substeps.VerticalCwtGaitEvents`
    - :meth:`skdh.gaitv3.substeps.CreateStridesAndQc`
    - :meth:`skdh.gaitv3.substeps.TurnDetection`
    """
    # gait parameters
    _params = [
        # event level endpoints
        gait_endpoints.StrideTime,
        gait_endpoints.StanceTime,
        gait_endpoints.SwingTime,
        gait_endpoints.StepTime,
        gait_endpoints.InitialDoubleSupport,
        gait_endpoints.TerminalDoubleSupport,
        gait_endpoints.DoubleSupport,
        gait_endpoints.SingleSupport,
        gait_endpoints.StepLengthModel1,
        gait_endpoints.StepLengthModel2,
        gait_endpoints.StrideLength,
        gait_endpoints.GaitSpeed,
        gait_endpoints.Cadence,
        gait_endpoints.IntraStepCovarianceV,
        gait_endpoints.IntraStrideCovarianceV,
        gait_endpoints.HarmonicRatioV,
        gait_endpoints.StrideSPARC,
        # bout level endpoints
        gait_endpoints.PhaseCoordinationIndex,
        gait_endpoints.GaitSymmetryIndex,
        gait_endpoints.StepRegularityV,
        gait_endpoints.StrideRegularityV,
        gait_endpoints.AutocovarianceSymmetryV,
        gait_endpoints.RegularityIndexV,
    ]

    def __init__(
            self,
            downsample=False,
            height_factor=0.53,
            provide_leg_length=False,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            gait_event_method='AP CWT',
            correct_orientation=True,
            filter_cutoff=20.0,
            filter_order=4,
            use_cwt_scale_relation=True,
            wavelet_scale='default',
            max_stride_time=lambda x: 2.0 * x + 1.0,
            loading_factor=lambda x: 0.17 * x + 0.05,
            bout_processing_pipeline=None,
    ):
        super().__init__(
            downsample=downsample,
            height_factor=height_factor,
            provide_leg_length=provide_leg_length,
            min_bout_time=min_bout_time,
            max_bout_separation_time=max_bout_separation_time,
            gait_event_method=gait_event_method,
            correct_orientation=correct_orientation,
            filter_cutoff=filter_cutoff,
            filter_order=filter_order,
            use_cwt_scale_relation=use_cwt_scale_relation,
            wavelet_scale=wavelet_scale,
            max_stride_time=max_stride_time,
            loading_factor=loading_factor,
            bout_processing_pipeline=bout_processing_pipeline,
        )

        self.downsample = downsample

        if provide_leg_length:
            self.height_factor = 1.0
        else:
            self.height_factor = height_factor

        self.min_bout_time = min_bout_time,
        self.max_bout_sep_time = max_bout_separation_time

        if bout_processing_pipeline is None:
            self.bout_pipeline = Pipeline(flatten_results=True)
            self.bout_pipeline.add(substeps.PreprocessGaitBout(
                correct_orientation=correct_orientation,
                filter_cutoff=filter_cutoff,
                filter_order=filter_order,
            ))
            if gait_event_method.lower() == "ap cwt":
                self.bout_pipeline.add(substeps.ApCwtGaitEvents())
            elif gait_event_method.lower() == "vertical cwt":
                self.bout_pipeline.add(substeps.VerticalCwtGaitEvents(
                    use_cwt_scale_relation=use_cwt_scale_relation,
                    wavelet_scale=wavelet_scale
                ))
            self.bout_pipeline.add(substeps.CreateStridesAndQc(
                max_stride_time=max_stride_time,
                loading_factor=loading_factor,
            ))
            self.bout_pipeline.add(substeps.TurnDetection())
        else:
            if isinstance(bout_processing_pipeline, Pipeline):
                self.bout_pipeline = bout_processing_pipeline
            else:
                raise ValueError("`bout_processing_pipeline` must be a `skdh.Pipeline` object.")

    def add_endpoints(self, endpoints):
        """
        Add endpoints to be computed

        Parameters
        ----------
        endpoints : {Iterable, GaitEventEndpoint, GaitBoutEndpoint}
            Either an iterable of GaitEventEndpoint or GaitBoutEndpoint references
            or an individual GaitEventEndpoint/GaitBoutEndpoint reference to be added
            to the list of endpoints to be computed.

        Examples
        --------
        >>> class NewGaitEndpoint(gait_endpoints.GaitEventEndpoint):
        >>>     pass
        >>>
        >>> gait = GaitV3()
        >>> gait.add_endpoints(NewGaitEndpoint)

        >>> class NewGaitEndpoint(gait_endpoints.GaitEventEndpoint):
        >>>     pass
        >>> class NewGaitEndpoint2(gait_endpoints.GaitEventEndpoint):
        >>>     pass
        >>>
        >>> gait = GaitV3()
        >>> gait.add_endpoints([NewGaitEndpoint, NewGaitEndpoint2])
        """
        if isinstance(endpoints, Iterable):
            if all(isinstance(i(), (GaitEventEndpoint, GaitBoutEndpoint)) for i in endpoints):
                self._params.extend(endpoints)
            else:
                raise ValueError("Not all objects are GaitEventEndpoints or GaitBoutEndpoints")
        else:
            if isinstance(endpoints(), (GaitEventEndpoint, GaitBoutEndpoint)):
                self._params.append(endpoints)
            else:
                raise ValueError(
                    f"Endpoint {endpoints!r} is not a GaitEventEndpoint or GaitBoutEndpoint"
                )

    @staticmethod
    def _handle_input_gait_predictions(gait_bouts, gait_pred, n_exp):
        """
        Handle gait predictions and broadcast to the correct type for the
        rest of the predict functionality.

        Parameters
        ----------
        gait_bouts : numpy.ndarray, optional
            (N, 2) array of gait starts (column 1) and stops (column 2). Either this
            or `gait_pred` is required in order to have gait analysis be performed
            on the data. `gait_bouts` takes precedence over `gait_pred`.
        gait_pred : {any, numpy.ndarray}, optional
            (N, ) array of boolean predictions of gait, or any value that is not
            None. If not an ndarray but not None, the entire recording will be
            taken as gait. If not provided (or None), gait classification will
            be performed on the acceleration data.
        n_exp : int
            Number of expected samples.

        Returns
        -------
        gait_pred_corr : numpy.ndarray
            Array of gait start and stop values shape(N, 2).
        """
        # handle gait bouts first as it has priority
        if gait_bouts is not None:
            bout_starts = gait_bouts[:, 0]
            bout_stops = gait_bouts[:, 1]

            return bout_starts, bout_stops

        if gait_pred is None:
            raise ValueError("One of `gait_bouts` or `gait_pred` must not be None.")
        elif isinstance(gait_pred, ndarray):
            if gait_pred.size != n_exp:
                raise ValueError("gait_pred shape does not match time & accel")
            lengths, starts, vals = rle(gait_pred.astype(int))

            bout_starts = starts[vals == 1]
            bout_stops = bout_starts + lengths[vals == 1]
        else:
            bout_starts = asarray([0])
            bout_stops = asarray([n_exp - 1])

        return bout_starts, bout_stops

    @handle_process_returns(results_to_kwargs=False)
    def predict(
            self,
            time=None,
            accel=None,
            *,
            gyro=None,
            fs=None,
            height=None,
            gait_bouts=None,
            gait_pred=True,
            v_axis=None,
            ap_axis=None,
            **kwargs,
    ):
        """
        predict(time, accel, *, gyro=None, fs=None, height=None, gait_pred=None, v_axis=None, ap_axis=None, day_ends={})

        Get the gait events and endpoints from a time series signal

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds
        accel : numpy.ndarray
            (N, 3) array of accelerations measured by a centrally mounted lumbar
            inertial measurement device, in units of 'g'.
        gyro : numpy.ndarray
            (N, 3) array of angular velocities measured by a centrally mounted
            lumbar inertial measurement device, in units of 'rad/s'. If provided,
            will be used to indicate if steps occur during turns. Default is None.
        fs : float, optional
            Sampling frequency in Hz of the accelerometer data. If not provided,
            will be computed form the timestamps.
        height : float, optional
            Either height (False) or leg length (True) of the subject who wore
            the inertial measurement device, in meters, depending on `leg_length`.
            If not provided, spatial endpoints will not be computed.
        gait_bouts : numpy.ndarray, optional
            (N, 2) array of gait starts (column 1) and stops (column 2). Either this
            or `gait_pred` is required in order to have gait analysis be performed
            on the data. `gait_bouts` takes precedence over `gait_pred`.
        gait_pred : {any, numpy.ndarray}, optional
            (N, ) array of boolean predictions of gait, or any value that is not
            None. If not a ndarray but not None, the entire recording will be
            taken as gait. If not provided (or None), gait classification will
            be performed on the acceleration data. Default is True.
        v_axis : {None, 0, 1, 2}, optional
            Vertical axis index. Default is None, which indicates it will be estimated
            from the acceleration data each gait bout.
        ap_axis : {None, 0, 1, 2}, optional
            AP axis index. Default is None, which indicates that it will be estimated
            from the acceleration data each bout.
        day_ends : dict, optional
            Optional dictionary containing (N, 2) arrays of start and stop
            indices for invididual days. Dictionary keys are in the format
            ({base}, {period}). If not provided, or the key specified by `day_window`
            is not found, no day-based windowing will be done.

        Returns
        -------
        gait_results : dict
            The computed gait endpoints. For a list of endpoints and their
            definitions, see :ref:`event-level-gait-endpoints` and
            :ref:`bout-level-gait-endpoints`.

        Raises
        ------
        ValueError
            If both `gait_bouts` and `gait_pred` are None.
        LowFrequencyError
            If the sampling frequency is less than 20Hz

        Notes
        -----
        Axis estimation
        ^^^^^^^^^^^^^^^
        The vertical axis is estimated as the axis with the highest absolute
        average acceleration during a gait bout. Since the acceleromter should be
        approximately aligned with the anatomical axes, this is a fairly easy estimation
        to perform

        The AP axis estimation is a little trickier, and depending on how the observed
        accelerometer wearer was walking, the AP and ML axes can be confused. The AP
        axis is estimated by first applying a butterworth filter (4th order, 3.0hz cutoff)
        to the acceleration data, and then computing the auto-covariance function 10 seconds
        or the bout length, whichever is shorter. The axis that has the closest autocorrelation
        with the vertical axis is then chosen as the AP axis.

        """
        super().predict(
            expect_days=True,
            expect_wear=False,  # currently not using wear
            time=time,
            accel=accel,
            gyro=gyro,
            fs=fs,
            height=height,
            gait_bouts=gait_bouts,
            gait_pred=gait_pred,
            **kwargs,
        )

        if height is None:
            warn("height not provided, not computing spatial metrics", UserWarning)
            leg_length = None
        else:
            # height factor is set to 1 if providing leg length
            leg_length = self.height_factor * height

        # compute fs/delta t if necessary
        fs = 1 / mean(diff(time)) if fs is None else fs
        if fs < 20.0:
            raise LowFrequencyError(f"Frequency ({fs:.2f}Hz) is too low (<20Hz).")

        # handle gait_pred input types
        gait_starts, gait_stops = self._handle_input_gait_predictions(
            gait_bouts, gait_pred, time.size)

        # get alternative names, that will be overwritten if downsampling
        goal_fs = fs
        time_rs = time
        accel_rs = accel
        gyro_rs = gyro
        gait_starts_rs = gait_starts
        gait_stops_rs = gait_stops
        day_starts_rs, day_stops_rs = self.day_idx

        if self.downsample:
            goal_fs = 50.0 if fs >= 50.0 else 20.0
            if fs != goal_fs:
                (
                    time_rs,
                    (accel_rs, gyro_rs),
                    (gait_starts_rs, gait_stops_rs, day_starts_rs, day_stops_rs),
                ) = apply_downsample(
                    goal_fs,
                    time,
                    (accel, gyro),
                    (gait_starts, gait_stops, *self.day_idx),
                    aa_filter=True,  # always want the AA filter for downsampling
                    fs=fs,
                )

        # setup the storage for the gait parameters
        gait = {
            i: []
            for i in [
                "Day N",
                "Bout N",
                "Bout Starts",
                "Bout Duration",
                "Bout Steps",
                "Gait Cycles",
                "IC",
                "FC",
                "FC opp foot",
                "forward cycles",
                # "delta h",  # handled in gait endpoints now
                "IC Time",
                "debug:mean step freq",
                "debug:v axis est",
                "debug:ap axis est",
                "Turn",
            ]
        }
        # aux dictionary for storing values for computing gait endpoints
        gait_aux = {
            i: []
            for i in [
                "v axis",
                "ap_axis",
                "accel",
                "inertial data i",
            ]
        }

        # keep track of where everything is in the loops
        gait_i = 0

        for iday, (dstart, dstop) in enumerate(zip(day_starts_rs, day_stops_rs)):
            # GET GAIT BOUTS
            gait_bouts = get_gait_bouts(
                gait_starts_rs,
                gait_stops_rs,
                dstart,
                dstop,
                time_rs,
                self.max_bout_sep_time,
                self.min_bout_time,
            )

            for ibout, bout in enumerate(gait_bouts):
                # run the bout processing pipeline
                bout_res = self.bout_pipeline.run(
                    time=time_rs[bout],
                    accel=accel_rs[bout],
                    gyro=gyro_rs[bout] if gyro_rs is not None else None,
                    fs=goal_fs,
                    v_axis=v_axis,
                    ap_axis=ap_axis,
                )

                # get the data we need
                n_strides = bout_res['qc_initial_contacts'].size
                gait['IC'].extend(bout_res["qc_initial_contacts"])
                gait['FC'].extend(bout_res['qc_final_contacts'])
                gait['FC opp foot'].extend(bout_res['qc_final_contacts_oppfoot'])
                gait['forward cycles'].extend(bout_res['forward cycles'])
                # optional
                gait['Turn'].extend(bout_res.get('step_in_turn', [-1] * n_strides))
                gait['debug:mean step freq'].extend([bout_res.get("mean_step_freq", nan)] * n_strides)
                gait['debug:v axis est'].extend([bout_res.get("v_axis_est", -1)] * n_strides)
                gait['debug:ap axis est'].extend([bout_res.get("ap_axis_est", -1)] * n_strides)

                # metadata
                gait['Bout N'].extend([ibout + 1] * n_strides)
                gait['Bout Starts'].extend([time_rs[bout.start]] * n_strides)
                gait['Bout Duration'].extend([(bout.stop - bout.start) / goal_fs] * n_strides)

                gait['Bout Steps'].extend([n_strides] * n_strides)
                gait['Gait Cycles'].extend([sum(bout_res['forward cycles'] == 2)])

                # gait auxiliary data
                gait_aux['accel'].append(bout_res.get("accel_filt", accel_rs[bout]))
                # add the index for the corresponding accel
                gait_aux['inertial data i'].extend(
                    [len(gait_aux['accel']) - 1] * n_strides
                )
                gait_aux['v axis'].extend([bout_res['v_axis']] * n_strides)
                gait_aux['ap axis'].extend([bout_res['ap_axis']] * n_strides)

            # add the day number
            gait['Day N'].extend(
                [iday + 1] * (len(gait['Bout N']) - len(gait['Day N']))
            )

        # convert to arrays
        for key in gait:
            gait[key] = asarray(gait[key])

        # convert some gait aux data to arrays
        gait_aux['inertial data i'] = asarray(gait_aux['inertial data i'])
        gait_aux['v axis'] = asarray(gait_aux['v axis'])
        gait_aux['ap axis'] = asarray(gait_aux['ap axis'])

        # loop over endpoints and compute if there is data to compute on
        if gait_aux['inertial data i'].size != 0:
            for param in self._params:
                param().predict(fs=goal_fs, leg_length=leg_length, gait=gait, gait_aux=gait_aux)

        # remove unnecessary stuff from the gait dict
        gait.pop("IC", None)
        gait.pop("FC", None)
        gait.pop("FC opp foot", None)
        gait.pop("forward cycles", None)

        return gait
