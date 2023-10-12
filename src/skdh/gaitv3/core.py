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

from skdh.gait.gait_endpoints import GaitEventEndpoint, GaitBoutEndpoint
from skdh.gait.gait_endpoints import gait_endpoints


class LowFrequencyError(Exception):
    pass


class GaitV3(BaseProcess):
    """
    Process lumbar IMU data to extract metrics of gait. Detect gait, extract gait
    events (heel-strikes, toe-offs), and compute gait metrics from inertial data
    collected from a lumbar mounted wearable inertial measurement unit. If angular
    velocity data is provided, turns are detected, and steps during turns are noted.

    Parameters
    ----------
    height_factor : float, optional
        The factor multiplied by height to obtain an estimate of leg length.
        Default is 0.53 [4]_. Ignored if `leg_length` is `True`
    provide_leg_length : bool, optional
        If the actual leg length will be provided. Setting to true would have the same effect
        as setting height_factor to 1.0 while providing leg length. Default is False
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
            height_factor=0.53,
            provide_leg_length=False
    ):
        super().__init__(
            height_factor=height_factor,
            provide_leg_length=provide_leg_length,
        )

        if provide_leg_length:
            self.height_factor = 1.0
        else:
            self.height_factor = height_factor

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
    def _handle_input_gait_predictions(gait_pred, n_exp):
        """
        Handle gait predictions and broadcast to the correct type for the
        rest of the predict functionality.

        Parameters
        ----------
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
        if gait_pred is None:
            return None, None
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
            gait_pred=None,
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
        gait_pred : {any, numpy.ndarray}, optional
            (N, ) array of boolean predictions of gait, or any value that is not
            None. If not an ndarray but not None, the entire recording will be
            taken as gait. If not provided (or None), gait classification will
            be performed on the acceleration data.
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
        if fs < 50.0:
            warn(
                f"Frequency ({fs:.2f}Hz) is less than 50Hz. Downsampling to 20Hz. Note "
                f"that this may effect gait metrics results values.",
                UserWarning,
            )

        # handle gait_pred input types
        gait_starts, gait_stops = self._handle_input_gait_predictions(gait_pred, time.size)

        goal_fs = 50.0 if fs >= 50.0 else 20.0
        if fs != goal_fs:
            (
                time_ds,
                (accel_ds, gyro_ds),
                (gait_starts_ds, gait_stops_ds, day_starts_ds, day_stops_ds),
            ) = apply_downsample(
                goal_fs,
                time,
                (accel, gyro),
                (gait_starts, gait_stops, *self.day_idx),
                aa_filter=True,  # always want the AA filter for downsampling
                fs=fs,
            )
        else:
            time_ds = time
            accel_ds = accel
            gyro_ds = gyro
            gait_starts_ds = gait_starts
            gait_stops_ds = gait_stops
            day_starts_ds, day_stops_ds = self.day_idx

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

        # get the gait classification if necessary
        gbout_starts, gbout_stops = get_gait_classification_lgbm(
            gait_starts_ds, gait_stops_ds, accel_ds, goal_fs,
        )

