"""
Gait detection, processing, and analysis from wearable inertial sensor data

Lukas Adamowicz
Copyright (c) 2023, Pfizer Inc. All rights reserved.
"""
from collections.abc import Iterable
from warnings import warn

from numpy import ndarray, asarray, mean, diff

from skdh.base import BaseProcess
from skdh.utility.internal import rle, apply_downsample
from skdh.utility.exceptions import LowFrequencyError

from skdh.gait.gait_endpoints import GaitEventEndpoint, GaitBoutEndpoint
from skdh.gait.gait_endpoints import gait_endpoints
from skdh.gaitv3.utility import get_gait_bouts


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
        gait_endpoints.StepLength,
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
    ):
        super().__init__(
            downsample=downsample,
            height_factor=height_factor,
            provide_leg_length=provide_leg_length,
            min_bout_time=min_bout_time,
            max_bout_separation_time=max_bout_separation_time,
        )

        self.downsample = downsample

        if provide_leg_length:
            self.height_factor = 1.0
        else:
            self.height_factor = height_factor

        self.min_bout_time = min_bout_time,
        self.max_bout_sep_time = max_bout_separation_time

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
            **kwargs,
    ):
        """
        predict(time, accel, *, gyro=None, fs=None, height=None, gait_pred=None, day_ends={})

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
                "delta h",
                "IC Time",
                "Turn",
            ]
        }
        # aux dictionary for storing values for computing gait endpoints
        gait_aux = {
            i: []
            for i in [
                "vert axis",
                "accel",
                "vert velocity",
                "vert position",
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
                # Step 1: signal processing, accelerometer orientation correction
                # Step 2: cadence/mean step time estimation
                # Step 3: gait event estimation
                # step 4: stride creation
                # step 5: turn detection
                # step 6: delta h estimation

