"""
Gait bout detection from accelerometer data

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import mean, diff, round
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d
from warnings import warn
from pathlib import Path
from importlib import resources
import lightgbm as lgb


from PfyMU.base import _BaseProcess
from PfyMU.utility import get_windowed_view
from PfyMU.features import Bank


def get_lgb_gait_classification(accel, fs)
    """
    Get classification of windows of accelerometer data

    Paramters
    ---------
    accel : numpy.ndarray
        (N, 3) array of acceleration values, in units of "g"
    fs : float
        Sampling frequency of the data
    """
    goal_fs = 50.0  # goal fs for classifier
    wlen = int(goal_fs * 3)
    step = int(0.75 * wlen)
    thresh = 0.7  # mean + 1 standard deviation of best threshold for maximizing F1 score

    # down/up sample if necessary
    if (fs != goal_fs):
        if (fs < goal_fs):
            warn(f"fs ({fs:.2f}) is less than 50.0Hz. Upsampling to 50.0Hz")
        
        f_rs = interp1d(np.arange(0, accel.shape[0]) / fs, 1 / fs, accel, axis=0, bounds_error=False, fill_value='extrapolate')

        accel_rs = f_rs(np.arange(0, accel.shape[0] / fs, 1 / goal_fs))
    else:
        accel_rs = accel
    
    # band-pass filter
    sos = butter(1, [2 * 0.25 / fs, 2 * 5 / fs], btype='band', output='sos')
    accel_rs_f = sosfiltfilt(sos, np.linalg.norm(accel_rs, axis=1))

    # window
    accel_w = get_windowed_view(accel_rs_f, wlen, wstep, ensure_c_contiguity=True)

    # get the feature bank
    feat_bank = Bank(window_length=None, window_step=None)  # make sure no windowing
    with resources.path('PfyMU.gait.models', 'final_features.json') as file_path:
        feat_bank.load(file_path)
    
    # compute the features
    accel_feats = feat_bank.compute(accel_w, fs=goal_fs, windowed=True)

    # load the model
    with resources.path('PfyMU.gait.models', 'lgbm_gait_classifier_no-stairs.lgbm') as file_path:
        bst = lgb.Booster(model_file=file_path)
    
    # predict
    gait_predictions = bst.predict(accel_feats)
    
    # expand the predictions to be per sample.
    tmp = np.zeros(accel.shape[0])

    c = np.zeros(accel.shape[0], dtype='int')  # keep track of overlap
    for i, p in enumerate(gait_predictions):
        i1 = i * wstep
        i2 = i1 + wlen
        # add prediction and keep track of counts effecting that cell to deal with
        # overlapping windows and therefore overlapping predictions
        tmp[i1:i2] += p
        c[i1:i2] += 1
    # make sure no divide by 0
    c[i2:] = 1

    tmp /= c
    gait_pred_ps = tmp >= 0.5  # final gait predictions
    return gait_pred_ps




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





