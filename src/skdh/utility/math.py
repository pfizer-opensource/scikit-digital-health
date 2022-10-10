"""
Utility math functions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import moveaxis, ascontiguousarray, full, nan

from skdh.utility import _extensions
from skdh.utility.windowing import get_windowed_view


__all__ = [
    "moving_mean",
    "moving_sd",
    "moving_skewness",
    "moving_kurtosis",
    "moving_median",
    "moving_max",
    "moving_min",
]


def moving_mean(a, w_len, skip, trim=True, axis=-1):
    r"""
    Compute the moving mean.

    Parameters
    ----------
    a : array-like
        Signal to compute moving mean for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving mean along. Default is -1.

    Returns
    -------
    mmean : numpy.ndarray
        Moving mean. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous.

    Notes
    -----
    On the moving axis if `trim=True`, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Most efficient computations are for `skip` values that are either factors of
    `wlen`, or greater or equal to `wlen`.

    Warnings
    --------
    Catastropic cancellation is a concern when `skip` is less than `wlen` due to
    the cumulative sum-type algorithm being used, when input values are very very
    large, or very very small. With typical IMU data values this should not be an
    issue, even for very long data series (multiple days worth of data)

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)
    >>> moving_mean(x, 3, 3)
    array([1., 4., 7.])

    Compute with overlapping windows:

    >>> moving_mean(x, 3, 1)
    array([1., 2., 3., 4., 5., 6., 7., 8.])

    Compute without trimming the result

    >>> moving_mean(x, 3, 1, trim=False)
    array([1., 2., 3., 4., 5., 6., 7., 8., nan, nan])

    Compute on a nd-array to see output shape. On the moving axis, the output
    should be equal to :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_mean(y, window_length, window_skip, axis=1)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_mean(z, 3, 3, axis=0).flags['C_CONTIGUOUS']
    False

    >>> moving_mean(z, 3, 3, axis=1).flags['C_CONTIGUOUS']
    False

    >>> moving_mean(z, 3, 3, axis=2).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # move computation axis to end
    x = moveaxis(a, axis, -1)

    # check that there are enough samples
    if w_len > x.shape[-1]:
        raise ValueError("Window length is larger than the computation axis.")

    rmean = _extensions.moving_mean(x, w_len, skip, trim)

    # move computation axis back to original place and return
    return moveaxis(rmean, -1, axis)


def moving_sd(a, w_len, skip, trim=True, axis=-1, return_previous=True):
    r"""
    Compute the moving sample standard deviation.

    Parameters
    ----------
    a : array-like
        Signal to compute moving sample standard deviation for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving mean along. Default is -1.
    return_previous : bool, optional
        Return previous moments. These are computed either way, and are therefore optional returns.
        Default is True.

    Returns
    -------
    msd : numpy.ndarray
        Moving sample standard deviation. Note that if the moving axis is not the last axis,
        then the result will *not* be c-contiguous.
    mmean : numpy.ndarray, optional.
        Moving mean. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous. Only returned if `return_previous=True`.

    Notes
    -----
    On the moving axis, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Most efficient computations are for `skip` values that are either factors of `wlen`, or greater
    or equal to `wlen`.

    Warnings
    --------
    Catastropic cancellation is a concern when `skip` is less than `wlen` due to the cumulative
    sum-type algorithms being used, when input values are very very large, or very very small. With
    typical IMU data values this should not be an issue, even for very long data series (multiple
    days worth of data).

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)**2
    >>> moving_sd(x, 3, 3, return_previous=True)
    (array([ 2.081666  ,  8.02080628, 14.0118997 ]),
     array([ 1.66666667, 16.66666667, 49.66666667]))

    Compute with overlapping windows:

    >>> moving_mean(x, 3, 1, return_previous=False)
    array([ 2.081666  ,  4.04145188,  6.02771377,  8.02080628, 10.0166528 ,
           12.01388086, 14.0118997 , 16.01041328])

    Compute without trimming:

    >>> moving_mean(x, 3, 1, trim=False, return_previous=False)
    array([ 2.081666  ,  4.04145188,  6.02771377,  8.02080628, 10.0166528 ,
           12.01388086, 14.0118997 , 16.01041328, nan, nan])

    Compute on a nd-array to see output shape. On the moving axis, the output should be equal to
    :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_sd(y, window_length, window_skip, axis=1, return_previous=False)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_sd(z, 3, 3, axis=0, return_previous=False).flags['C_CONTIGUOUS']
    False

    >>> moving_sd(z, 3, 3, axis=1, return_previous=False).flags['C_CONTIGUOUS']
    False

    >>> moving_sd(z, 3, 3, axis=2, return_previous=False).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # move computation axis to end
    x = moveaxis(a, axis, -1)

    # check that there are enough samples
    if w_len > x.shape[-1]:
        raise ValueError(
            "Cannot have a window length larger than the computation axis."
        )

    res = _extensions.moving_sd(x, w_len, skip, trim, return_previous)

    # move computation axis back to original place and return
    if return_previous:
        return moveaxis(res[0], -1, axis), moveaxis(res[1], -1, axis)
    else:
        return moveaxis(res, -1, axis)


def moving_skewness(a, w_len, skip, trim=True, axis=-1, return_previous=True):
    r"""
    Compute the moving sample skewness.

    Parameters
    ----------
    a : array-like
        Signal to compute moving skewness for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving mean along. Default is -1.
    return_previous : bool, optional
        Return previous moments. These are computed either way, and are therefore optional returns.
        Default is True.

    Returns
    -------
    mskew : numpy.ndarray
        Moving skewness. Note that if the moving axis is not the last axis,
        then the result will *not* be c-contiguous.
    msd : numpy.ndarray, optional
        Moving sample standard deviation. Note that if the moving axis is not the last axis,
        then the result will *not* be c-contiguous. Only returned if `return_previous=True`.
    mmean : numpy.ndarray, optional.
        Moving mean. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous. Only returned if `return_previous=True`.

    Notes
    -----
    On the moving axis, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Warnings
    --------
    While this implementation is quite fast, it is also quite mememory inefficient. 3 arrays
    of equal length to the computation axis are created during computation, which can easily
    exceed system memory if already using a significant amount of memory.

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)**2
    >>> moving_skewness(x, 3, 3, return_previous=True)
    (array([0.52800497, 0.15164108, 0.08720961]),
     array([ 2.081666  ,  8.02080628, 14.0118997 ]),
     array([ 1.66666667, 16.66666667, 49.66666667]))

    Compute with overlapping windows:

    >>> moving_skewness(x, 3, 1, return_previous=False)
    array([0.52800497, 0.29479961, 0.20070018, 0.15164108, 0.12172925,
           0.10163023, 0.08720961, 0.07636413])

    Compute without trimming:

    >>> moving_skewness(x, 3, 1, trim=False, return_previous=False)
    array([0.52800497, 0.29479961, 0.20070018, 0.15164108, 0.12172925,
           0.10163023, 0.08720961, 0.07636413, nan, nan])

    Compute on a nd-array to see output shape. On the moving axis, the output should be equal to
    :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_skewness(y, window_length, window_skip, axis=1, return_previous=False)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_skewness(z, 3, 3, axis=0, return_previous=False).flags['C_CONTIGUOUS']
    False

    >>> moving_skewness(z, 3, 3, axis=1, return_previous=False).flags['C_CONTIGUOUS']
    False

    >>> moving_skewness(z, 3, 3, axis=2, return_previous=False).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # move computation axis to end
    x = moveaxis(a, axis, -1)

    # check that there are enough samples
    if w_len > x.shape[-1]:
        raise ValueError(
            "Cannot have a window length larger than the computation axis."
        )

    res = _extensions.moving_skewness(x, w_len, skip, trim, return_previous)

    # move computation axis back to original place and return
    if return_previous:
        return tuple(moveaxis(i, -1, axis) for i in res)
    else:
        return moveaxis(res, -1, axis)


def moving_kurtosis(a, w_len, skip, trim=True, axis=-1, return_previous=True):
    r"""
    Compute the moving sample kurtosis.

    Parameters
    ----------
    a : array-like
        Signal to compute moving kurtosis for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving mean along. Default is -1.
    return_previous : bool, optional
        Return previous moments. These are computed either way, and are therefore optional returns.
        Default is True.

    Returns
    -------
    mkurt : numpy.ndarray
        Moving kurtosis. Note that if the moving axis is not the last axis,
        then the result will *not* be c-contiguous.
    mskew : numpy.ndarray, optional
        Moving skewness. Note that if the moving axis is not the last axis,
        then the result will *not* be c-contiguous. Only returned if `return_previous=True`.
    msd : numpy.ndarray, optional
        Moving sample standard deviation. Note that if the moving axis is not the last axis,
        then the result will *not* be c-contiguous. Only returned if `return_previous=True`.
    mmean : numpy.ndarray, optional.
        Moving mean. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous. Only returned if `return_previous=True`.

    Notes
    -----
    On the moving axis, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Warnings
    --------
    While this implementation is quite fast, it is also quite mememory inefficient. 4 arrays
    of equal length to the computation axis are created during computation, which can easily
    exceed system memory if already using a significant amount of memory.

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)**2
    >>> moving_kurtosis(x, 3, 3, return_previous=True)
    (array([-1.5, -1.5, -1.5]),  # kurtosis
     array([0.52800497, 0.15164108, 0.08720961]),  # skewness
     array([ 2.081666  ,  8.02080628, 14.0118997 ]),  # standard deviation
     array([ 1.66666667, 16.66666667, 49.66666667]))  # mean

    Compute with overlapping windows:

    >>> moving_kurtosis(np.random.random(100), 50, 20, return_previous=False)
    array([-1.10155074, -1.20785479, -1.24363625])  # random

    Compute without trimming:

    >>> moving_kurtosis(np.random.random(100), 50, 20, return_previous=False)
    array([-1.10155074, -1.20785479, -1.24363625, nan, nan])  # random

    Compute on a nd-array to see output shape. On the moving axis, the output should be equal to
    :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_skewness(y, window_length, window_skip, axis=1, return_previous=False)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_kurtosis(z, 3, 3, axis=0, return_previous=False).flags['C_CONTIGUOUS']
    False

    >>> moving_kurtosis(z, 3, 3, axis=1, return_previous=False).flags['C_CONTIGUOUS']
    False

    >>> moving_kurtosis(z, 3, 3, axis=2, return_previous=False).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # move computation axis to end
    x = moveaxis(a, axis, -1)

    # check that there are enough samples
    if w_len > x.shape[-1]:
        raise ValueError(
            "Cannot have a window length larger than the computation axis."
        )

    res = _extensions.moving_kurtosis(x, w_len, skip, trim, return_previous)

    # move computation axis back to original place and return
    if return_previous:
        return tuple(moveaxis(i, -1, axis) for i in res)
    else:
        return moveaxis(res, -1, axis)


def moving_median(a, w_len, skip=1, trim=True, axis=-1):
    r"""
    Compute the moving mean.

    Parameters
    ----------
    a : array-like
        Signal to compute moving mean for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples. Default is 1.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving mean along. Default is -1.

    Returns
    -------
    mmed : numpy.ndarray
        Moving median. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous.

    Notes
    -----
    On the moving axis, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)
    >>> moving_median(x, 3, 3)
    array([1., 4., 7.])

    Compute with overlapping windows:

    >>> moving_median(x, 3, 1)
    array([1., 2., 3., 4., 5., 6., 7., 8.])

    Compute without trimming:

    >>> moving_median(x, 3, 1)
    array([1., 2., 3., 4., 5., 6., 7., 8., nan, nan])

    Compute on a nd-array to see output shape. On the moving axis, the output should be equal to
    :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_median(y, window_length, window_skip, axis=1)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_median(z, 3, 3, axis=0).flags['C_CONTIGUOUS']
    False

    >>> moving_median(z, 3, 3, axis=1).flags['C_CONTIGUOUS']
    False

    >>> moving_median(z, 3, 3, axis=2).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # move computation axis to end
    x = moveaxis(a, axis, -1)

    # check that there are enough samples
    if w_len > x.shape[-1]:
        raise ValueError(
            "Cannot have a window length larger than the computation axis."
        )

    rmed = _extensions.moving_median(x, w_len, skip, trim)

    # move computation axis back to original place and return
    return moveaxis(rmed, -1, axis)


def moving_max(a, w_len, skip, trim=True, axis=-1):
    r"""
    Compute the moving maximum value.

    Parameters
    ----------
    a : array-like
        Signal to compute moving max for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving max along. Default is -1.

    Returns
    -------
    mmax : numpy.ndarray
        Moving max. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous.

    Notes
    -----
    On the moving axis, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)
    >>> moving_max(x, 3, 3)
    array([2., 5., 8.])

    Compute with overlapping windows:

    >>> moving_max(x, 3, 1)
    array([2., 3., 4., 5., 6., 7., 8.])

    Compute without triming:

    >>> moving_max(x, 3, 1)
    array([2., 3., 4., 5., 6., 7., 8., nan, nan])

    Compute on a nd-array to see output shape. On the moving axis, the output should be equal to
    :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_max(y, window_length, window_skip, axis=1)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_max(z, 3, 3, axis=0).flags['C_CONTIGUOUS']
    False

    >>> moving_max(z, 3, 3, axis=1).flags['C_CONTIGUOUS']
    False

    >>> moving_max(z, 3, 3, axis=2).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # Numpy uses SIMD instructions for max/min, so it will likely be faster
    # unless there is a lot of overlap
    cond1 = a.ndim == 1 and (skip / w_len) < 0.005
    cond2 = a.ndim > 1 and (skip / w_len) < 0.3  # due to c-contiguity?
    cond3 = a.ndim > 2  # windowing doesnt handle more than 2 dimensions currently
    if any([cond1, cond2, cond3]):
        # move computation axis to end
        x = moveaxis(a, axis, -1)

        # check that there are enough samples
        if w_len > x.shape[-1]:
            raise ValueError("Window length is larger than the computation axis.")

        rmax = _extensions.moving_max(x, w_len, skip, trim)

        # move computation axis back to original place and return
        return moveaxis(rmax, -1, axis)
    else:
        x = ascontiguousarray(
            moveaxis(a, axis, 0)
        )  # need to move axis to the front for windowing
        xw = get_windowed_view(x, w_len, skip)
        if trim:
            res = xw.max(axis=1)  # computation axis is still the second axis
        else:
            nfill = (x.shape[0] - w_len) // skip + 1
            rshape = list(x.shape)
            rshape[0] = (x.shape[0] - 1) // skip + 1
            res = full(rshape, nan)
            res[:nfill] = xw.max(axis=1)

        return moveaxis(res, 0, axis)


def moving_min(a, w_len, skip, trim=True, axis=-1):
    r"""
    Compute the moving maximum value.

    Parameters
    ----------
    a : array-like
        Signal to compute moving max for.
    w_len : int
        Window length in number of samples.
    skip : int
        Window start location skip in number of samples.
    trim : bool, optional
        Trim the ends of the result, where a value cannot be calculated. If False,
        these values will be set to NaN. Default is True.
    axis : int, optional
        Axis to compute the moving max along. Default is -1.

    Returns
    -------
    mmax : numpy.ndarray
        Moving max. Note that if the moving axis is not the last axis, then the result
        will *not* be c-contiguous.

    Notes
    -----
    On the moving axis, the output length can be computed as follows:

    .. math:: \frac{n - w_{len}}{skip} + 1

    where `n` is the length of the moving axis. For cases where `skip != 1` and
    `trim=False`, the length of the return on the moving axis can be calculated as:

    .. math:: \frac{n}{skip}

    Examples
    --------
    Compute the with non-overlapping windows:

    >>> import numpy as np
    >>> x = np.arange(10)
    >>> moving_min(x, 3, 3)
    array([1., 4., 7.])

    Compute with overlapping windows:

    >>> moving_min(x, 3, 1)
    array([1., 2., 3., 4., 5., 6., 7.])

    Compute without trimming:

    >>> moving_min(x, 3, 1)
    array([1., 2., 3., 4., 5., 6., 7., nan, nan])


    Compute on a nd-array to see output shape. On the moving axis, the output should be equal to
    :math:`(n - w_{len}) / skip + 1`.

    >>> n = 500
    >>> window_length = 100
    >>> window_skip = 50
    >>> shape = (3, n, 5, 10)
    >>> y = np.random.random(shape)
    >>> res = moving_min(y, window_length, window_skip, axis=1)
    >>> print(res.shape)
    (3, 9, 5, 10)

    Check flags for different axis output

    >>> z = np.random.random((10, 10, 10))
    >>> moving_min(z, 3, 3, axis=0).flags['C_CONTIGUOUS']
    False

    >>> moving_min(z, 3, 3, axis=1).flags['C_CONTIGUOUS']
    False

    >>> moving_min(z, 3, 3, axis=2).flags['C_CONTIGUOUS']
    True
    """
    if w_len <= 0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # Numpy uses SIMD instructions for max/min, so it will likely be faster
    # unless there is a lot of overlap
    cond1 = a.ndim == 1 and (skip / w_len) < 0.005
    cond2 = a.ndim > 1 and (skip / w_len) < 0.3  # due to c-contiguity?
    cond3 = a.ndim > 2  # windowing doesnt handle more than 2 dimensions currently
    if any([cond1, cond2, cond3]):
        # move computation axis to end
        x = moveaxis(a, axis, -1)

        # check that there are enough samples
        if w_len > x.shape[-1]:
            raise ValueError("Window length is larger than the computation axis.")

        rmin = _extensions.moving_min(x, w_len, skip, trim)

        # move computation axis back to original place and return
        return moveaxis(rmin, -1, axis)
    else:
        x = ascontiguousarray(
            moveaxis(a, axis, 0)
        )  # need to move axis to the front for windowing
        xw = get_windowed_view(x, w_len, skip)
        if trim:
            res = xw.min(axis=1)  # computation axis is still the second axis
        else:
            nfill = (x.shape[0] - w_len) // skip + 1
            rshape = list(x.shape)
            rshape[0] = (x.shape[0] - 1) // skip + 1
            res = full(rshape, nan)
            res[:nfill] = xw.min(axis=1)

        return moveaxis(res, 0, axis)
