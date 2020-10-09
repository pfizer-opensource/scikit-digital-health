"""
Utility methods

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import require
from numpy.lib.stride_tricks import as_strided


def get_windowed_view(x, window_length, step_size, ensure_c_contiguity=False):
    """
    Return a moving window view over the data
    Parameters
    ----------
    x : numpy.ndarray
        1- or 2-D array of signals to window. Windows occur along the 0 axis.
        Must be C-contiguous.
    window_length : int
        Window length/size.
    step_size : int
        Step/stride size for windows - how many samples to step from window
        center to window center.
    ensure_c_contiguity : bool, optional
        Create a new array with C-contiguity if the passed array is not C-contiguous.
        This *may* result in the memory requirements significantly increasing. Default is False,
        which will raise a ValueError if `x` is not C-contiguous
    Returns
    -------
    x_win : numpy.ndarray
        2- or 3-D array of windows of the original data, of shape (..., L[, ...])
    """
    if not (x.ndim in [1, 2]):
        raise ValueError('Array cannot have more than 2 dimensions.')

    if ensure_c_contiguity:
        x = require(x, requirements=['C'])
    else:
        if not x.flags['C_CONTIGUOUS']:
            raise ValueError("Input array must be C-contiguous.  See numpy.ascontiguousarray")

    if x.ndim == 1:
        nrows = ((x.size - window_length) // step_size) + 1
        n = x.strides[0]
        return as_strided(
            x,
            shape=(nrows, window_length),
            strides=(step_size * n, n),
            writeable=False
        )

    else:
        k = x.shape[1]
        nrows = ((x.shape[0] - window_length) // step_size) + 1
        n = x.strides[1]

        new_shape = (nrows, window_length, k)
        new_strides = (step_size * k * n, k * n, n)
        return as_strided(x, shape=new_shape, strides=new_strides, writeable=False)
