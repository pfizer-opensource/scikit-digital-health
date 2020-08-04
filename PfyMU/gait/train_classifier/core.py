"""
Methods for loading data and training classifiers for gait
"""
from pathlib import Path
import h5py
from scipy.interpolate import interp1d
import numpy as np

from PfyMU.features.utility import get_windowed_view, compute_window_samples


__all__ = ['load_datasets']


def load_datasets(paths, device_location=None, goal_fs=100.0, acc_mag=True, window_length=3.0, window_step=0.5, signal_function=None):
    """
    Load standardized datasets into memory

    Parameters
    ----------
    paths : str, Path, array_like
        Path to a dataset or an array-like of strings for multiple datasets
    device_location : {str, None}, optional
        Body location of the device. None indicates that no body location should be looked for in the data files.
    goal_fs : float, optional
        Desired sampling frequency. Will interpolate up/down depending on the sampling frequency of the
        provided datasets. Default is 100.0
    acc_mag : bool, optional
        Compute and use the acceleration magnitude instead of the 3-axis acceleration. Default is True.
    window_length : float
        Window length in seconds. If not provided (None), will do no windowing. Default is 3.0 seconds
    window_step : {float, int, dict}
        Window step - the spacing between the start of windows. This can be specified several different ways
        (see Notes). Default is 0.5. If providing a dictionary, each key is an activity, and its value
        is the value of the overlap. You can specify a default overlap by using the key 'default'.
    signal_function : None, function
        Function to apply to the data (or data magnitude). Signature is `function(signal, fs)`, and it should return
        `transformed_signal`. If a function is provided, but `acc_mag=True`, then
        the function is applied after taking the magnitude.

    Returns
    -------
    dataset : numpy.ndarray
        (M, N, P) array of acceleration data windowed as specified. M is the total number of windows accross all
        datasets loaded. N is the window length, and P is the number of axes (3 for non-magnitude, 1 for using
        acceleration magnitude)
    labels : numpy.ndarray
        (M, ) array of labels as to whether the window contains gait data or not.
    subjects : numpy.ndarray
        (M, ) array of subject identifiers. In order to ensure that these are distinct between studies, the number of
        the study (determined by order in `paths`) is appended to the subject name, e.g. `subject1_3` would be
        `subject1` for the 3rd study in `paths`.
    activities : numpy.ndarray
        (M, ) array of the specific activity identifiers.

    Notes
    -----
    Computation of the window step depends on the type of input provided, and the range.
    - `window_step` is a float in (0.0, 1.0]: specifies the fraction of a window to skip to get to the start of the
    next window
    - `window_step` is an integer > 1: specifies the number of samples to skip to get to the start of the next
    window
    """
    # TODO add support for continuous data with time-varying labels
    # TODO add support for non-windowed data
    # TODO add support for gyroscope data
    # TODO add support for various other processing to the acceleration signals (ie filtering) before windowing

    if isinstance(paths, (str, Path)):
        paths = [paths]

    paths = [Path(i) for i in paths]  # make sure entries are Path objects

    if isinstance(window_step, dict):
        n_wstep = {}
        if 'default' not in window_step:
            window_step['default'] = 0.5
        for act in window_step:
            n_wlen, n_wstep[act] = compute_window_samples(goal_fs, window_length, window_step[act])
        step_d = True
    else:
        n_wlen, n_wstep = compute_window_samples(goal_fs, window_length, window_step)
        step_d = False

    M, N = 0, n_wlen
    
    # determine the last dimension of the output array after functions are applied
    if acc_mag:
        if signal_function is not None:
            try:
                P = signal_function(np.random.rand(50), goal_fs).shape[1]
            except IndexError:
                P = 1
        else:
            P = 1
    else:
        try:
            P = signal_function(np.random.rand(50, 3), goal_fs).shape[1]
        except IndexError:
            P = 1

    # first pass to get size for array allocation
    for dset in paths:
        # find all the subjects in the dataset
        subjs = [i for i in dset.glob('*.h5') if i.is_file()]

        for subj in subjs:
            with h5py.File(subj, 'r') as f:
                for activity in f.keys():
                    for trial in f[activity].keys():
                        if device_location is None:
                            n = f[activity][trial]['Accelerometer'].shape[0]
                            fs = f[activity][trial].attrs.get('Sampling rate')
                        else:
                            if device_location not in f[activity][trial]:
                                continue
                            n = f[activity][trial][device_location]['Accelerometer'].shape[0]
                            fs = f[activity][trial][device_location].attrs.get('Sampling rate')

                        n = int(np.ceil(n * goal_fs / fs))  # compute samples when down/upsampled
                        if n < n_wlen:
                            continue
                        if step_d:
                            M += int(((n - n_wlen) // n_wstep.get(activity, n_wstep['default']) + 1))
                        else:
                            M += int(((n - n_wlen) // n_wstep + 1))
    
    # allocate space for the data
    dataset = np.zeros((M, N) if (P == 1) else (M, N, P))
    subjects = np.empty(M, dtype='U30')  # maximum 30 character strings
    activities = np.empty(M, dtype='U30')  # maximum 30 character strings
    labels = np.empty(M, dtype='int')

    cnt = 0  # keeping track of index

    # second pass to get the data from the datasets
    for di, dset in enumerate(paths):
        # find all the subjects in the dataset
        subjs = [i for i in dset.glob('*.h5') if i.is_file()]
        for subj in subjs:
            with h5py.File(subj, 'r') as f:
                for activity in f.keys():
                    gait_label = f[activity].attrs.get('Gait label')
                    for trial in f[activity].keys():
                        if device_location is None:
                            fsloc = f'{activity}/{trial}'
                            loc = fsloc + '/Accelerometer'
                        else:
                            if device_location not in f[activity][trial]:
                                continue
                            fsloc = f'{activity}/{trial}/{device_location}'
                            loc = fsloc + '/Accelerometer'
                        
                        n = f[loc].shape[0]
                        fs = f[fsloc].attrs.get('Sampling rate')

                        # ensure there is enough data
                        if n < (n_wlen * fs / goal_fs):
                            continue

                        if fs != goal_fs:
                            f_intrp = interp1d(
                                np.arange(0, n/fs, 1/fs)[:n],
                                f[loc],
                                axis=0,
                                bounds_error=False,
                                fill_value='extrapolate'
                            )
                            tmp = f_intrp(np.arange(0, n/fs, 1/goal_fs))
                        else:
                            tmp = f[loc][()]

                        if acc_mag:
                            tmp = np.linalg.norm(tmp, axis=1)

                        if signal_function is not None:
                            tmp = np.ascontiguousarray(signal_function(tmp, goal_fs))

                        if step_d:
                            m = int(((tmp.shape[0] - n_wlen) // n_wstep.get(activity, n_wstep['default']) + 1))
                            dataset[cnt:cnt + m] = get_windowed_view(tmp, n_wlen, n_wstep.get(activity, n_wstep['default']))
                        else:
                            m = int(((tmp.shape[0] - n_wlen) // n_wstep + 1))
                            dataset[cnt:cnt + m] = get_windowed_view(tmp, n_wlen, n_wstep)

                        # append study/dataset number to seperate studies
                        subjects[cnt:cnt+m] = f'{subj.name.split(".")[0]}_{di}'
                        activities[cnt:cnt+m] = activity
                        labels[cnt:cnt+m] = gait_label

                        cnt += m  # increment count

    return dataset, labels, subjects, activities
