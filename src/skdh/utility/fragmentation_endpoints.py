"""
Generic endpoints dealing with the fragmentation of binary predictions.

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from numpy import (
    mean,
    asarray,
    cumsum,
    minimum,
    sort,
    argsort,
    unique,
    insert,
    sum,
    log,
    nan,
    float_,
)

from skdh.utility.internal import rle


__all__ = [
    "average_duration",
    "state_transition_probability",
    "gini_index",
    "average_hazard",
    "state_power_law_distribution",
]


def gini(x, w=None, corr=True):
    """
    Compute the GINI Index.

    Parameters
    ----------
    x : numpy.ndarray
        Array of bout lengths
    w : {None, numpy.ndarray}, optional
        Weights for x. Must be the same size. If None, weights are not used.
    corr : bool, optional
        Apply finite sample correction. Default is True.

    Returns
    -------
    g : float
        Gini index

    References
    ----------
    .. [1] https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in
        -python/48999797#48999797
    """
    if x.size == 0:
        return 0.0
    elif x.size == 1:
        return 1.0

    # The rest of the code requires numpy arrays.
    if w is not None:
        sorted_indices = argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = cumsum(sorted_w, dtype=float_)
        cumxw = cumsum(sorted_x * sorted_w, dtype=float_)
        g = sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1])
        if corr:
            return g * x.size / (x.size - 1)
        else:
            return g
    else:
        sorted_x = sort(x)
        n = x.size
        cumx = cumsum(sorted_x, dtype=float_)
        # The above formula, with all weights equal to 1 simplifies to:
        g = (n + 1 - 2 * sum(cumx) / cumx[-1]) / n
        if corr:
            return minimum(g * n / (n - 1), 1)
        else:
            return g


def average_duration(a=None, *, lengths=None, values=None, voi=1):
    """
    Compute the average duration in the desired state.

    Parameters
    ----------
    a : array-like, optional
        1D array of binary values. If not provided, all of `lengths`, `starts`,
        and `values` must be provided.
    lengths : {numpy.ndarray, list}, optional
        Lengths of runs of the binary values. If not provided, `a` must be. Must
        be the same size as `values`.
    values : {numpy.ndarray, list}, optional
        Values of the runs. If not provided, all `lengths` will be assumed to be
        for the `voi`.
    voi : {int, bool}, optional
        Value of interest, value for which to calculate the average run length.
        Default is `1` (`True`).

    Returns
    -------
    avg_dur : float
        average duration, in samples, of the runs with value `voi`.

    Examples
    --------

    >>> x = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    >>> average_duration(x, voi=1)
    2.0
    >>> average_duration(x, voi=0)
    5.333333333

    >>> lengths = [4, 2, 9, 1, 3, 3]
    >>> values = [0, 1, 0, 1, 0, 1]
    >>> average_duration(lengths=lengths, values=values, voi=1)
    2.0
    >>> average_duration(lengths=lengths)
    2.0
    """
    if a is not None:
        l, _, v = rle(a)
        lens = l[v == voi]
    else:
        if lengths is None:
            raise ValueError("One of `a` or `lengths` must be provided.")
        lens = asarray(lengths)

        if values is not None:
            lens = lens[values == voi]

    if lens.size == 0:
        return 0.0

    return mean(lens)


def state_transition_probability(a=None, *, lengths=None, values=None, voi=1):
    r"""
    Compute the probability of transitioning from the desired state to the
    second state.

    Parameters
    ----------
    a : array-like, optional
        1D array of binary values. If not provided, all of `lengths`, `starts`,
        and `values` must be provided.
    lengths : {numpy.ndarray, list}, optional, optional
        Lengths of runs of the binary values. If not provided, `a` must be. Must
        be the same size as `values`.
    values : {numpy.ndarray, list}, optional, optional
        Values of the runs. If not provided, all `lengths` will be assumed to be
        for the `voi`.
    voi : {int, bool}, optional
        Value of interest, value for which to calculate the average run length.
        Default is `1` (`True`).

    Returns
    -------
    avg_dur : float
        average duration, in samples, of the runs with value `voi`.

    Examples
    --------

    >>> x = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    >>> state_transition_probability(x, voi=1)
    0.5
    >>> state_transition_probability(x, voi=0)
    0.1875

    >>> lengths = [4, 2, 9, 1, 3, 3]
    >>> values = [0, 1, 0, 1, 0, 1]
    >>> state_transition_probability(lengths=lengths, values=values, voi=1)
    0.5
    >>> state_transition_probability(lengths=lengths)
    0.5

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate more frequent switching between states, and as a result may indicate
    greater fragmentation of sleep.

    The implementation is straightforward [1]_, and is simply defined as

    .. math:: satp = \frac{1}{\mu_{awake}}

    where :math:`\mu_{awake}` is the mean awake bout time.
    """
    if a is not None:
        l, _, v = rle(a)
        lens = l[v == voi]
    else:
        if lengths is None:
            raise ValueError("One of `a` or `lengths` must be provided.")
        lens = asarray(lengths)

        if values is not None:
            lens = lens[values == voi]

    if lens.size == 0:
        return nan

    return 1 / mean(lens)


def gini_index(a=None, *, lengths=None, values=None, voi=1):
    """
    Compute the normalized variability of the state bouts, also known as the GINI
    index from economics.

    Parameters
    ----------
    a : array-like, optional
        1D array of binary values. If not provided, all of `lengths`, `starts`,
        and `values` must be provided.
    lengths : {numpy.ndarray, list}, optional, optional
        Lengths of runs of the binary values. If not provided, `a` must be. Must
        be the same size as `values`.
    values : {numpy.ndarray, list}, optional, optional
        Values of the runs. If not provided, all `lengths` will be assumed to be
        for the `voi`.
    voi : {int, bool}, optional
        Value of interest, value for which to calculate the average run length.
        Default is `1` (`True`).

    Returns
    -------
    avg_dur : float
        average duration, in samples, of the runs with value `voi`.

    Examples
    --------

    >>> x = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    >>> gini_index(x, voi=1)
    0.333333
    >>> gini_index(x, voi=0)
    0.375

    >>> lengths = [4, 2, 9, 1, 3, 3]
    >>> values = [0, 1, 0, 1, 0, 1]
    >>> gini_index(lengths=lengths, values=values, voi=1)
    0.333333
    >>> gini_index(lengths=lengths)
    0.333333

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Gini Index values are bounded between 0 and 1, with values near 1 indicating the total
    time accumulating due to a small number of longer bouts, whereas values near 0 indicate all
    bouts contribute more equally to the total time.
    """
    if a is not None:
        l, _, v = rle(a)
        lens = l[v == voi]
    else:
        if lengths is None:
            raise ValueError("One of `a` or `lengths` must be provided.")
        lens = asarray(lengths)

        if values is not None:
            lens = lens[values == voi]

    if lens.size == 0:
        return 0.0

    return gini(lens, w=None, corr=True)


def average_hazard(a=None, *, lengths=None, values=None, voi=1):
    r"""
    Compute the average hazard summary of the hazard function, as a function of the
    state bout duration. The average hazard represents a summary of the frequency
    of transitioning from one state to the other.

    Parameters
    ----------
    a : array-like, optional
        1D array of binary values. If not provided, all of `lengths`, `starts`,
        and `values` must be provided.
    lengths : {numpy.ndarray, list}, optional, optional
        Lengths of runs of the binary values. If not provided, `a` must be. Must
        be the same size as `values`.
    values : {numpy.ndarray, list}, optional, optional
        Values of the runs. If not provided, all `lengths` will be assumed to be
        for the `voi`.
    voi : {int, bool}, optional
        Value of interest, value for which to calculate the average run length.
        Default is `1` (`True`).

    Returns
    -------
    avg_dur : float
        average duration, in samples, of the runs with value `voi`.

    Examples
    --------

    >>> x = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    >>> average_hazard(x, voi=1)
    0.61111111
    >>> average_hazard(x, voi=0)
    0.61111111

    >>> lengths = [4, 2, 9, 1, 3, 3]
    >>> values = [0, 1, 0, 1, 0, 1]
    >>> average_hazard(lengths=lengths, values=values, voi=1)
    0.61111111
    >>> average_hazard(lengths=lengths)
    0.61111111

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv,
        p. 182337, Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate higher frequency in switching from sleep to awake states.

    The average hazard is computed per [1]_:

    .. math::

        h(t_n_i) = \frac{n\left(t_n_i\right)}{n - n^c\left(t_n_{i-1}\right)}
        \har{h} = \frac{1}{m}\sum_{t\in D}h(t)

    where :math:`h(t_n_i)` is the hazard for the sleep bout of length :math:`t_n_i`,
    :math:`n(t_n_i)` is the number of bouts of length :math:`t_n_i`, :math:`n` is
    the total number of sleep bouts, :math:`n^c(t_n_i)` is the sum number of bouts
    less than or equal to length :math:`t_n_i`, and :math:`t\in D` indicates all
    bouts up to the maximum length (:math:`D`).
    """
    if a is not None:
        l, _, v = rle(a)
        lens = l[v == voi]
    else:
        if lengths is None:
            raise ValueError("One of `a` or `lengths` must be provided.")
        lens = asarray(lengths)

        if values is not None:
            lens = lens[values == voi]

    if lens.size == 0:
        return nan

    unq, cnts = unique(lens, return_counts=True)
    sidx = argsort(unq)

    cnts = cnts[sidx]
    cumsum_cnts = insert(cumsum(cnts), 0, 0)

    h = cnts / (cumsum_cnts[-1] - cumsum_cnts[:-1])

    return sum(h) / unq.size


def state_power_law_distribution(a=None, *, lengths=None, values=None, voi=1):
    r"""
    Compute the scaling factor for the power law distribution over the desired
    state bout lengths.

    Parameters
    ----------
    a : array-like, optional
        1D array of binary values. If not provided, all of `lengths`, `starts`,
        and `values` must be provided.
    lengths : {numpy.ndarray, list}, optional, optional
        Lengths of runs of the binary values. If not provided, `a` must be. Must
        be the same size as `values`.
    values : {numpy.ndarray, list}, optional, optional
        Values of the runs. If not provided, all `lengths` will be assumed to be
        for the `voi`.
    voi : {int, bool}, optional
        Value of interest, value for which to calculate the average run length.
        Default is `1` (`True`).

    Returns
    -------
    avg_dur : float
        average duration, in samples, of the runs with value `voi`.

    Examples
    --------

    >>> x = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    >>> state_power_law_distribution(x, voi=1)
    1.7749533004219864
    >>> state_power_law_distribution(x, voi=0)
    2.5517837760569524

    >>> lengths = [4, 2, 9, 1, 3, 3]
    >>> values = [0, 1, 0, 1, 0, 1]
    >>> state_power_law_distribution(lengths=lengths, values=values, voi=1)
    1.7749533004219864
    >>> state_power_law_distribution(lengths=lengths)
    1.7749533004219864

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv,
        p. 182337, Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Larger `alpha` values indicate that the total sleeping time is accumulated with
    a larger portion of shorter sleep bouts.

    The power law scaling factor is computer per [1]_:

    .. math:: 1 + \frac{n_{sleep}}{\sum_{i}\log{t_i / \left(min(t) - 0.5\right)}}

    where :math:`n_{sleep}` is the number of sleep bouts, :math:`t_i` is the duration
    of the :math:`ith` sleep bout, and :math:`min(t)` is the length of the shortest
    sleep bout.
    """
    if a is not None:
        l, _, v = rle(a)
        lens = l[v == voi]
    else:
        if lengths is None:
            raise ValueError("One of `a` or `lengths` must be provided.")
        lens = asarray(lengths)

        if values is not None:
            lens = lens[values == voi]

    if lens.size == 0:
        return 1.0

    return 1 + lens.size / sum(log(lens / (lens.min() - 0.5)))
