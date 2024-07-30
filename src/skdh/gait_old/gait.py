"""
Gait detection, processing, and analysis from wearable inertial sensor data

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from sys import gettrace
from collections.abc import Iterable
from warnings import warn
from pathlib import Path

import h5py
from numpy import mean, diff, asarray, sum, ndarray
from numpy.linalg import norm
from pandas import Timestamp
import matplotlib
import matplotlib.pyplot as plt

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility.internal import apply_resample, rle
from skdh.utility.exceptions import LowFrequencyError

from skdh.gait_old.get_gait_classification import (
    get_gait_classification_lgbm,
)
from skdh.gait_old.get_gait_bouts import get_gait_bouts
from skdh.gait_old.get_gait_events import get_gait_events
from skdh.gait_old.get_strides import get_strides
from skdh.gait_old.get_turns import get_turns
from skdh.gait_old.gait_endpoints import gait_endpoints
from skdh.gait_old.gait_endpoints import GaitEventEndpoint, GaitBoutEndpoint


class Gait(BaseProcess):
    """
    Process IMU data to extract endpoints of gait. Detect gait, extract gait events
    (heel-strikes, toe-offs), and compute gait endpoints from inertial data collected
    from a lumbar mounted wearable inertial measurement unit. If angular velocity
    data is provided, turns are detected, and steps during turns are noted.

    .. deprecated:: 0.13.0
        `skdh.gait.Gait` has been superseded by `skdh.gaitv3.LumbarGait`

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
        relationship.
    wavelet_scale : {"default", float, int}, optional
        The wavelet scale to use. If `use_cwt_scale_relation=True`, then this is only
        used initially to determine the optimal scale. If `False`, then is used as the
        scale for the initial and final contact event detection. `"default"`
        corresponds to the default scale from [3]_, scaled for the sampling frequency.
        If a float, this is the value in Hz that the desired wavelet decomposition
        happens. For reference, [3]_ used a frequency of 1.25Hz. If an integer,
        uses that value as the scale.
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
    downsample_aa_filter : bool, optional
        Apply an anti-aliasing filter before downsampling. Default is True.
        Uses the same IIR filter as :py:func:`scipy.signal.decimate`.
    day_window : array-like
        Two (2) element array-like of the base and period of the window to use for determining
        days. Default is (0, 24), which will look for days starting at midnight and lasting 24
        hours. None removes any day-based windowing.

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

    If angular velocity data is provided, turns are detected [8]_, and steps during
    turns are noted in the results. Values are assigned as follows:

    - -1: Turns not detected (lacking angular velocity data)
    - 0: No turn found
    - 1: Turn overlaps with either Initial or Final contact
    - 2: Turn overlaps with both Initial and Final contact

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
    .. [8] M. H. Pham et al., “Algorithm for Turning Detection and Analysis
        Validated under Home-Like Conditions in Patients with Parkinson’s Disease
        and Older Adults using a 6 Degree-of-Freedom Inertial Measurement Unit at
        the Lower Back,” Front. Neurol., vol. 8, Apr. 2017,
        doi: 10.3389/fneur.2017.00135.
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
        correct_accel_orient=True,
        use_cwt_scale_relation=True,
        wavelet_scale="default",
        min_bout_time=8.0,
        max_bout_separation_time=0.5,
        max_stride_time=2.25,
        loading_factor=0.2,
        height_factor=0.53,
        prov_leg_length=False,
        filter_order=4,
        filter_cutoff=20.0,
        downsample_aa_filter=True,
        day_window=(0, 24),
    ):
        super().__init__(
            # key-word arguments for storage
            correct_accel_orient=correct_accel_orient,
            use_cwt_scale_relation=use_cwt_scale_relation,
            wavelet_scale=wavelet_scale,
            min_bout_time=min_bout_time,
            max_bout_separation_time=max_bout_separation_time,
            max_stride_time=max_stride_time,
            loading_factor=loading_factor,
            height_factor=height_factor,
            prov_leg_length=prov_leg_length,
            filter_order=filter_order,
            filter_cutoff=filter_cutoff,
            downsample_aa_filter=downsample_aa_filter,
            day_window=day_window,
        )

        warn(
            "Deprecated in version 0.13.0. Use `skdh.gaitv3.LumbarGait` instead.",
            DeprecationWarning,
        )

        self.corr_accel_orient = correct_accel_orient
        self.use_opt_scale = use_cwt_scale_relation
        self.cwt_scale = wavelet_scale
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

        self.aa_filter = downsample_aa_filter

        # for saving gait predictions
        self._save_classifier_fn = lambda time, starts, stops: None

        # is plotting available/valid
        self.valid_plot = False
        # enable plotting as a public method
        self.setup_plotting = self._setup_plotting

        if day_window is None:
            self.day_key = (-1, -1)
        else:
            self.day_key = tuple(day_window)

    def _save_classifier_predictions(self, fname):  # pragma: no cover
        def fn(time, starts, stops):
            with h5py.File(fname, "w") as f:
                f["time"] = time
                f["bout starts"] = starts
                f["bout stops"] = stops

        self._save_classifier_fn = fn

    def add_endpoints(self, endpoints):
        """
        Add endpoints to be computed

        Parameters
        ----------
        endpoints : {Iterable, callable}
            Either an iterable of GaitEventEndpoint or GaitBoutEndpoint references or an individual
            GaitEventEndpoint/GaitBoutEndpoint reference to be added to the list of endpoints to be
            computed.

        Examples
        --------
        >>> class NewGaitEndpoint(gait_endpoints.GaitEventEndpoint):
        >>>     pass
        >>>
        >>> gait = Gait()
        >>> gait.add_endpoints(NewGaitEndpoint)

        >>> class NewGaitEndpoint(gait_endpoints.GaitEventEndpoint):
        >>>     pass
        >>> class NewGaitEndpoint2(gait_endpoints.GaitEventEndpoint):
        >>>     pass
        >>>
        >>> gait = Gait()
        >>> gait.add_endpoints([NewGaitEndpoint, NewGaitEndpoint2])
        """
        if isinstance(endpoints, Iterable):
            if all(
                isinstance(i(), (GaitEventEndpoint, GaitBoutEndpoint))
                for i in endpoints
            ):
                self._params.extend(endpoints)
            else:
                raise ValueError(
                    "Not all objects are GaitEventEndpoints or GaitBoutEndpoints."
                )
        else:
            if isinstance(endpoints(), (GaitEventEndpoint, GaitBoutEndpoint)):
                self._params.append(endpoints)
            else:
                raise ValueError(
                    f"Endpoint {endpoints!r} is not a GaitEventEndpoints or "
                    f"GaitBoutEndpoints"
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

    def _handle_wavelet_scale(self, fs):
        """
        Compute the scale to use for the wavelet decompositions.

        Parameters
        ----------
        fs : float
            Sampling frequency.

        Returns
        -------
        scale : int
            Wavelet decomposition scale.
        """
        # original scale. Compute outside loop since stays the same
        # 1.25 comes from original paper, corresponds to desired frequency
        # 0.2 is the central frequency of the 'gaus1' wavelet (normalized to 1)
        original_scale = max(round(0.2 / (1.25 / fs)), 1)

        if self.cwt_scale == "default":
            scale = original_scale
        elif isinstance(self.cwt_scale, float):
            scale = max(round(0.2 / (self.cwt_scale / fs)), 1)
        elif isinstance(self.cwt_scale, int):
            scale = self.cwt_scale
        else:
            raise ValueError(
                "Type of `wavelet_scale` [{type(self.cwt_scale)}] not understood."
            )

        return scale

    def _setup_plotting(self, save_file, debug=False):  # pragma: no cover
        """
        Setup gait specific plotting.

        Parameters
        ----------
        save_file : str
            The file name to save the resulting plot to. Extension will be set to
            PDF. There are formatting options as well for dynamically generated
            names. See Notes.

        Notes
        -----
        Available format variables available:

        - date: todays date expressed in yyyymmdd format.
        - name: process name.
        - file: file name used in the pipeline, or "" if not found.
        """
        if save_file is None:
            return

        if gettrace() is None and not debug:  # only set if not debugging
            matplotlib.use("PDF")
            # non-interactive, don't want to be displaying plots constantly
        plt.style.use("ggplot")

        self.plot_fname = save_file

    @handle_process_returns(results_to_kwargs=False)
    def predict(
        self,
        *,
        time,
        accel,
        gyro=None,
        fs=None,
        height=None,
        gait_pred=None,
        **kwargs,
    ):
        """
        predict(*, time, accel, gyro=None, fs=None, height=None, gait_pred=None, day_ends={})

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
        warn(
            "Deprecated in version 0.13.0. Use `skdh.gaitv3.LumbarGait` instead.",
            DeprecationWarning,
        )

        if height is None:
            warn("height not provided, not computing spatial endpoints", UserWarning)
            leg_length = None
        else:
            # height factor is set to 1 if providing leg length
            leg_length = self.height_factor * height

        # compute fs/delta t
        fs = 1 / mean(diff(time)) if fs is None else fs
        if fs < 20.0:
            raise LowFrequencyError(f"Frequency ({fs:.2f}Hz) is too low (<20Hz).")
        if fs < 50:
            warn(
                "Frequency is less than 50Hz. Downsampling to 20Hz. Note that "
                "this may effect gait endpoints results values",
                UserWarning,
            )

        # check if plotting is available (<10min of data)
        self.valid_plot = (time[-1] - time[0]) < (10 * 60)
        self._initialize_plot(kwargs.get("file", self.plot_fname))

        # handle gait_pred input types
        gait_starts, gait_stops = self._handle_input_gait_predictions(
            gait_pred, time.size
        )

        goal_fs = 50.0 if fs >= 50.0 else 20.0
        if fs != goal_fs:
            (
                time_ds,
                (accel_ds, gyro_ds),
                (gait_starts_ds, gait_stops_ds, day_starts_ds, day_stops_ds),
            ) = apply_resample(
                goal_fs=goal_fs,
                time=time,
                data=(accel, gyro),
                indices=(gait_starts, gait_stops, *self.day_idx),
                aa_filter=self.aa_filter,
                fs=fs,
            )
        else:
            time_ds = time
            accel_ds = accel
            gyro_ds = gyro
            gait_starts_ds = gait_starts
            gait_stops_ds = gait_stops
            day_starts_ds, day_stops_ds = self.day_idx

        wavelet_scale = self._handle_wavelet_scale(goal_fs)

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
            gait_starts_ds, gait_stops_ds, accel_ds, goal_fs
        )
        self._save_classifier_fn(time_ds, gbout_starts, gbout_stops)

        gait_i = 0  # keep track of where everything is in the loops

        for iday, (start, stop) in enumerate(zip(day_starts_ds, day_stops_ds)):
            # GET GAIT BOUTS
            # ==============
            gait_bouts = get_gait_bouts(
                gbout_starts,
                gbout_stops,
                start,
                stop,
                time_ds,
                self.max_bout_sep,
                self.min_bout,
            )

            for ibout, bout in enumerate(gait_bouts):
                # get the gait events, vertical acceleration, and vertical axis
                ic, fc, vert_acc, v_axis = get_gait_events(
                    accel_ds[bout],
                    goal_fs,
                    time_ds[bout],
                    wavelet_scale,
                    self.filt_ord,
                    self.filt_cut,
                    self.corr_accel_orient,
                    self.use_opt_scale,
                )

                # get the strides
                strides_in_bout = get_strides(
                    gait,
                    vert_acc,
                    gait_i,
                    ic,
                    fc,
                    time_ds[bout],
                    goal_fs,
                    self.max_stride_time,
                    self.loading_factor,
                )

                # check if strides are during turns
                get_turns(
                    gait,
                    accel_ds[bout],
                    gyro_ds[bout] if gyro_ds is not None else None,
                    goal_fs,
                    strides_in_bout,
                )

                # plotting
                self._plot(time_ds, accel_ds, bout, ic, fc, gait, strides_in_bout)

                # add inertial data to the aux dict for use in gait endpoints calculation
                gait_aux["accel"].append(accel_ds[bout, :])
                # add the index for the corresponding accel/velocity/position
                gait_aux["inertial data i"].extend(
                    [len(gait_aux["accel"]) - 1] * strides_in_bout
                )
                gait_aux["vert axis"].extend([v_axis] * strides_in_bout)

                # save some default per bout endpoints
                gait["Bout N"].extend([ibout + 1] * strides_in_bout)
                gait["Bout Starts"].extend([time_ds[bout.start]] * strides_in_bout)
                gait["Bout Duration"].extend(
                    [(bout.stop - bout.start) / goal_fs] * strides_in_bout
                )

                gait["Bout Steps"].extend([strides_in_bout] * strides_in_bout)
                gait["Gait Cycles"].extend(
                    [sum(asarray(gait["forward cycles"][gait_i:]) == 2)]
                    * strides_in_bout
                )

                gait_i += strides_in_bout

            # add the day number
            gait["Day N"].extend(
                [iday + 1] * (len(gait["Bout N"]) - len(gait["Day N"]))
            )

        # convert to arrays
        for key in gait:
            gait[key] = asarray(gait[key])
        # convert inertial data index to an array
        gait_aux["inertial data i"] = asarray(gait_aux["inertial data i"])
        gait_aux["vert axis"] = asarray(gait_aux["vert axis"])

        # loop over endpoints and compute if there is data to compute on
        if len(gait_aux["inertial data i"]) != 0:
            for param in self._params:
                param().predict(goal_fs, leg_length, gait, gait_aux)

        # finalize/save the plot
        self._finalize_plot(kwargs.get("file", self.plot_fname))

        # remove unnecessary stuff from gait dict
        gait.pop("IC", None)
        gait.pop("FC", None)
        gait.pop("FC opp foot", None)
        gait.pop("forward cycles", None)

        return gait

    def _initialize_plot(self, file):  # pragma: no cover
        """
        Setup the plot
        """
        if self.valid_plot and self.plot_fname is not None:
            fname = Path(file).name if file is not None else "file-None"

            self.f, self.ax = plt.subplots(figsize=(12, 5))
            self.f.suptitle(f"Gait Visual Report: {fname}")

            self.ax.set_xlabel(r"Time [$s$]")
            self.ax.set_ylabel(r"Accel. [$\frac{m}{s^2}$]")

    def _plot(self, time, accel, gait_bout, ic, fc, gait, sib):  # pragma: no cover
        if self.valid_plot and self.f is not None:
            rtime = time[gait_bout] - time[0]
            baccel = norm(accel[gait_bout], axis=1)

            self.ax.plot(rtime, baccel, label="Accel. Mag.", color="C0")
            self.ax.plot(rtime[ic], baccel[ic], "x", color="k", label="Poss. IC")
            self.ax.plot(rtime[fc], baccel[fc], "+", color="k", label="Poss. FC")

            # valid contacts
            allc = gait["IC"][-sib:] + gait["FC"][-sib:] + gait["FC opp foot"][-sib:]
            self.ax.plot(
                rtime[allc],
                baccel[allc],
                "o",
                color="g",
                alpha=0.4,
                label="Valid Contact",
            )

    def _finalize_plot(self, file):  # pragma: no cover
        if self.valid_plot and self.f is not None:
            date = Timestamp.today().strftime("%Y%m%d")
            form_fname = self.plot_fname.format(date=date, file=Path(file).stem)

            self.ax.legend(loc="best")
            self.f.tight_layout()

            self.f.savefig(Path(form_fname).with_suffix(".pdf"))
