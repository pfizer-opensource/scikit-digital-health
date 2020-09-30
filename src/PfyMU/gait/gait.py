"""
Gait detection, processing, and analysis from wearable inertial sensor data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn

from numpy import mean, std, diff, abs, argmax, argmin, arange, fft
from scipy.signal import detrend, butter, sosfiltfilt, find_peaks
from scipy.integrate import cumtrapz
from pywt import cwt, scale2frequency

from PfyMU.base import _BaseProcess
from PfyMU.gait.bout_detection import get_lgb_gait_classification


class Gait(_BaseProcess):
    def __init__(self, use_cwt_scale_relation=True, max_stride_time=2.25, loading_factor=0.2,
                 height_factor=0.53, leg_length=False, filter_order=4, filter_cutoff=20.0):
        """
        Detect gait, extract gait events (heel-strikes, toe-offs), and compute gait metrics from
        inertial data collected from a lumbar mounted wearable inertial measurement unit

        Parameters
        ----------
        use_cwt_scale_relation : bool, optional
            Use the optimal scale/frequency relationship defined in [5]_. This changes which
            scale is used for the smoothing/differentiation operation performed with the
            continuous wavelet transform. Default is True. See Notes for a caveat of the
            relationship
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
        :math:`loading_factor * max_stride_time`
        2. Stance time must be less than
        :math:`(max_stride_time/2) + loading_factor * max_stride_time`
        3. Stride time must be less than `max_stride_time`

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
        super().__init__('Gait Process')

        self.use_opt_scale = use_cwt_scale_relation

        self.max_stride_time = max_stride_time
        self.loading_factor = loading_factor

        self.height_factor = height_factor
        self.leg_length = leg_length

        self.filt_ord = filter_order
        self.filt_cut = filter_cutoff

    def _predict(self, *, time=None, accel=None, gyro=None, height=None, **kwargs):
        """
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

        Returns
        -------
        gait_results : dict
        """
        super()._predict(time=time, accel=accel, gyro=gyro, **kwargs)

        if 'height' is None:
            warn('height not provided, not computing spatial metrics', UserWarning)
        else:
            # if not providing leg length (default), multiply height by the height factor
            if not self.leg_length:
                height = self.height_factor * height

        # compute fs/delta t
        dt = mean(diff(time[:500]))

        # check if windows exist for days
        if self._days in kwargs:
            days = kwargs[self._days]
        else:
            days = [(0, accel.shape[0])]

        results = {}

        # get the gait classifications
        gait_class = get_lgb_gait_classification(accel, 1 / dt)

        # figure out vertical axis
        v_axis = argmax(abs(mean(accel[:500])))

        for iday, day_idx in enumerate(days):
            start, stop = day_idx

            # GET GAIT EVENTS
            # ======================================
            vert_acc = detrend(accel[start:stop, v_axis])

            # low-pass filter
            sos = butter(self.filt_ord, 2 * self.filt_cut * dt, btype='low', output='sos')
            fvert_acc = sosfiltfilt(sos, vert_acc)

            # first integrate the vertical acceleration to get vertical velocity
            vert_vel = cumtrapz(fvert_acc, dx=dt, initial=0)

            # differentiate using the continuous wavelet transform with a gaus1 wavelet
            coef1, freq = cwt(vert_vel, arange(1, 50), 'gaus1', sampling_period=dt)

            # 1.25 corresponds to the scale used in the original paper
            scale1 = argmin(abs(freq - 1.25))

            # if using the optimal scale relationship, get the optimal scale
            if self.use_opt_scale:
                F = abs(fft.rfft(coef1[scale1]))
                # compute an estimate of the step frequency
                step_freq = argmax(F) / coef1.shape[1] / dt

                ic_opt_freq = 0.69 * step_freq + 0.34
                fc_opt_freq = 3.6 * step_freq - 4.5

                scale1 = argmin(abs(freq - ic_opt_freq))
                scale2 = argmin(abs(freq - fc_opt_freq))
            else:
                scale2 = scale1

            """
            Find the peaks in the coefficients at the computed scale. 
            Normally, this should be the troughs in the signal, but the way PyWavelets computes
            the CWT results in an inverted signal, so finding peaks in the normal signal
            works as intended and matches the original papers
            """
            ic, *_ = find_peaks(coef1[scale1], height=0.5*std(coef1[scale1]))

            coef2, _ = cwt(coef1[scale2], scale2, 'gaus1')

            """
            Peaks are the final contact points
            This matches the expected result from the original papers
            """
            fc, *_ = find_peaks(coef2[0], height=0.5 * std(coef2[0]))

            # GET STEPS/STRIDES/ETC
            # ======================================
