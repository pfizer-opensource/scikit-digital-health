"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import mean, diff, round
from scipy.signal import butter, sosfiltfilt


from PfyMU.base import _BaseProcess
from PfyMU.utility import get_windowed_view


class ThresholdGaitDetection(_BaseProcess):
    def __init__(self, vertical_axis='y', ):
        super().__init__()

    def apply(self, *args):
        """
        Apply the threshold-based gait detection

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        accel : numpy.ndarray
            (N, 3) array of acceleration values, in g.
        angvel : {numpy.ndarray, None}
            (N, 3) array of angular velocity values, in rad/s, or None if not available.
        temperature : {numpy.ndarray, None}
            (N, ) array of temperature values, in deg. C, or None if not available.

        References
        ----------
        Hickey, A, S. Del Din, L. Rochester, A. Godfrey. "Detecting free-living steps and walking bouts: validating
        an algorithm for macro gait analysis." Physiological Measurement. 2016
        """
        time, accel, _, temp, *_ = args  # don't need angular velocity

        # determine sampling frequency from timestamps
        fs = 1 / mean(diff(time))  # TODO make this only some of the samples?

        # apply a 2nd order lowpass filter with a cutoff of 17Hz
        sos = butter(2, 17 / (0.5 * fs), btype='low', output='sos')
        accel_proc = sosfiltfilt(sos, accel, axis=0)

        # remove the axis means
        accel_proc = accel_proc - mean(accel_proc, axis=0, keepdims=True)

        # window in 0.1s non-overlapping windows
        n_0p1 = int(round(0.1 * fs))
        # set ensure_c_contiguity to True to make a copy of the array in the c format if necessary to create the windows
        accel_win = get_windowed_view(accel_proc, n_0p1, n_0p1, ensure_c_contiguity=True)

        # take the mean of the vertical axis





