"""
Methods for loading data and training classifiers for gait
"""
from pathlib import Path
import h5py
from scipy.interpolate import interp1d
import numpy as np

from PfyMU.features.utility import get_windowed_view, compute_window_samples


__all__ = ['load_datasets']


def load_datasets(paths, goal_fs=100.0, acc_mag=True, window_length=3.0, window_step=0.5):
    """
    Load standardized datasets into memory

    Parameters
    ----------
    paths : str, Path, array_like
        Path to a dataset or an array-like of strings for multiple datasets
    goal_fs : float, optional
        Desired sampling frequency. Will interpolate up/down depending on the sampling frequency of the
        provided datasets. Default is 100.0
    acc_mag : bool, optional
        Compute and use the acceleration magnitude instead of the 3-axis acceleration. Default is True.
    window_length : float
        Window length in seconds. If not provided (None), will do no windowing. Default is 3.0 seconds
    window_step : {float, int}
        Window step - the spacing between the start of windows. This can be specified several different ways
        (see Notes). Default is 0.5

    Returns
    -------
    dataset : numpy.ndarray
        (M, N, P) array of acceleration data windowed as specified. M is the total number of windows accross all
        datasets loaded. N is the window length, and P is the number of axes (3 for non-magnitude, 1 for using
        acceleration magnitude)
    subjects : numpy.ndarray
        (M, ) array of subject identifiers. In order to ensure that these are distinct between studies, the number of
        the study (determined by order in `paths`) is appended to the subject name, e.g. `subject1_3` would be
        `subject` for the 3rd study in `paths`.
    labels : numpy.ndarray
        (M, ) array of labels as to whether the window contains gait data or not.

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

    # compute the goal window length and the goal window step
    n_wlen, n_wstep = compute_window_samples(goal_fs, window_length, window_step)

    M, N = 0, n_wlen

    # first pass to get size for array allocation
    for dset in paths:
        # find all the subjects in the dataset
        subjs = [i for i in dset.glob('subject*.h5') if i.is_file()]

        for subj in subjs:
            with h5py.File(subj, 'r') as f:
                for activity in f.keys():
                    for trial in f[activity].keys():
                        n, _ = f[activity][trial]['Accelerometer'].shape
                        fs = f[activity][trial].attrs.get('Sampling rate')

                        n = int(n * goal_fs / fs)  # compute samples when down/upsampled
                        M += int((n - n_wlen) / n_wstep + 1)

    # allocate space for the data
    dataset = np.zeros((M, N)) if acc_mag else np.zeros((M, N, 3))
    subjects = np.empty(M, dtype='U30')  # maximum 30 character strings
    labels = np.empty(M, dtype='int')

    cnt = 0  # keeping track of index

    # second pass to get the data from the datasets
    for di, dset in enumerate(paths):
        # find all the subjects in the dataset
        subjs = [i for i in dset.glob('subject*.h5') if i.is_file()]
        for subj in subjs:
            with h5py.File(subj, 'r') as f:
                # first pass to get size for array allocation
                for activity in f.keys():
                    gait_label = f[activity].attrs.get('Gait Label')
                    for trial in f[activity].keys():
                        n, _ = f[activity][trial]['Accelerometer'].shape
                        fs = f[activity][trial].attrs.get('Sampling rate')
                        if fs != goal_fs:
                            f_intrp = interp1d(np.arange(0, n/fs, 1/fs), f[activity][trial]['Accelerometer'], axis=0)
                            tmp = f_intrp(np.arange(0, n/fs, 1/goal_fs))
                        else:
                            tmp = f[activity][trial]

                        if acc_mag:
                            tmp = np.linalg.norm(tmp, axis=1)

                        m = int(((tmp.shape[0] - n_wlen) / n_wstep + 1))
                        dataset[cnt:cnt+m] = get_windowed_view(tmp, n_wlen, n_wstep)
                        subjects[cnt:cnt+m] = f'{subj.name}_{di}'  # append study/dataset number to seperate studies
                        labels[cnt:cnt+m] = gait_label

                        cnt += m  # increment count

    return dataset, subjects, labels



