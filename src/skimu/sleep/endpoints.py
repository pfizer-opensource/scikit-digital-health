"""
Sleep-based endpoints

Yiorgos Christakis, Lukas Adamowicz
Pfizer DMTI 2019-2021
"""
from numpy import around, nonzero, diff, argmax, sum, mean, log, unique, argsort, cumsum, insert, \
    int_

from skimu.sleep.utility import rle, gini


__all__ = [
    "total_sleep_time", "percent_time_asleep", "number_of_wake_bouts", "sleep_onset_latency",
    "wake_after_sleep_onset"
]


def total_sleep_time(sleep_predictions):
    """
    Compute the total time spent asleep, in minutes from sleep predictions.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    tst : int
        Total time spent asleep
    """
    return sum(sleep_predictions)


def percent_time_asleep(sleep_predictions):
    """
    Compute the percent time spent asleep from 1 minute epoch sleep predictions.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    pta : float
        Percent time asleep of the total sleep opportunity.
    """
    pta = 100.0 * sum(sleep_predictions) / sleep_predictions.size
    return around(pta, decimals=3)


def number_of_wake_bouts(sleep_predictions):
    """
    Compute the number of waking bouts during the total sleep opportunity, excluding the
    first wake before sleep, and last wake bout after sleep.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    nwb : int
        Number of waking bouts.
    """
    # -1 to exclude the last wakeup
    return nonzero(diff(sleep_predictions.astype(int_)) == -1)[0].size - 1


def sleep_onset_latency(sleep_predictions):
    """
    Compute the amount of time before the first sleep period in minutes.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    sol : int
        Total number of minutes spent awake before the first sleep period
    """
    return argmax(sleep_predictions)  # samples = minutes


def wake_after_sleep_onset(sleep_predictions):
    """
    Compute the number of minutes awake after the first period of sleep, excluding the last
    wake period after sleep.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    waso : int
        Total number of minutes spent awake after the first sleep period
    """
    first_epoch, last_epoch = nonzero(sleep_predictions)[0][[0, -1]]
    waso = (last_epoch - first_epoch) - sum(sleep_predictions[first_epoch:last_epoch])
    return waso


def average_sleep_duration(sleep_predictions):
    """
    Compute the average duration of a sleep bout.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    asp : float
        Average number of minutes per bout of sleep during total sleep opportunity.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate longer bouts of sleep.
    """
    lengths, starts, vals = rle(sleep_predictions)
    sleep_lengths = lengths[vals == 1]

    return mean(sleep_lengths)


def average_wake_duration(sleep_predictions):
    """
    Compute the average duration of wake bouts during sleep.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    awp : float
        Average number of minutes per bout of wake during total sleep opportunity.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate longer bouts of wakefulness.
    """
    lengths, starts, vals = rle(sleep_predictions)
    wake_lengths = lengths[vals == 0]

    return mean(wake_lengths)


def sleep_awake_transition_probability(sleep_predictions):
    r"""
    Compute the probability of transitioning from sleep state to awake state

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    satp : float
        Sleep to awake transition probability during the total sleep opportunity.

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

    .. math:: satp = \frac{1}{\mu_{sleep}}

    where :math:`\mu_{sleep}` is the mean sleep bout time.
    """
    lengths, starts, vals = rle(sleep_predictions)
    sleep_lengths = lengths[vals == 1]

    return 1 / mean(sleep_lengths)


def awake_sleep_transition_probability(sleep_predictions):
    r"""
    Compute the probability of transitioning from awake state to sleep state.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    astp : float
        Awake to sleep transition probability during the total sleep opportunity.

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
    lengths, starts, vals = rle(sleep_predictions)
    wake_lengths = lengths[vals == 0]

    return 1 / mean(wake_lengths)


def sleep_gini_index(sleep_predictions):
    """
    Compute the normalized variability of the sleep bouts, also known as the Gini Index from
    economics.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    gini : float
        Sleep normalized variability or Gini Index during total sleep opportunity.

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
    lengths, starts, vals = rle(sleep_predictions)
    sleep_lengths = lengths[vals == 1]

    return gini(sleep_lengths, w=None, corr=True)


def awake_gini_index(sleep_predictions):
    """
    Compute the normalized variability of the awake bouts, also known as the Gini Index from
    economics.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    gini : float
        Awake normalized variability or Gini Index during total sleep opportunity.

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
    lengths, starts, vals = rle(sleep_predictions)
    wake_lengths = lengths[vals == 0]

    return gini(wake_lengths, w=None, corr=True)


def sleep_average_hazard(sleep_predictions):
    r"""
    Compute the average hazard summary of the hazard function as a function of the sleep bout
    duration. The average hazard represents a summary of the frequency of transitioning from
    a sleep to awake state.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    h_sleep : float
        Sleep bout average hazard.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate higher frequency in switching from sleep to awake states.

    The average hazard is computed per [1]_:

    .. math::

        h(t_n_i) = \frac{n\left(t_n_i\right)}{n - n^c\left(t_n_{i-1}\right)}
        \har{h} = \frac{1}{m}\sum_{t\in D}h(t)

    where :math:`h(t_n_i)` is the hazard for the sleep bout of length :math:`t_n_i`,
    :math:`n(t_n_i)` is the number of bouts of length :math:`t_n_i`, :math:`n` is the total
    number of sleep bouts, :math:`n^c(t_n_i)` is the sum number of bouts less than or equal to
    length :math:`t_n_i`, and :math:`t\in D` indicates all bouts up to the maximum length
    (:math:`D`).
    """
    lengths, starts, vals = rle(sleep_predictions)
    sleep_lengths = lengths[vals == 1]

    u_sl, c_sl = unique(sleep_lengths, return_counts=True)
    sidx = argsort(u_sl)

    c_sl = c_sl[sidx]
    cs_c_sl = insert(cumsum(c_sl), 0, 0)

    h_i = c_sl / (cs_c_sl[-1] - cs_c_sl[:-1])

    return sum(h_i) / u_sl.size


def awake_average_hazard(sleep_predictions):
    r"""
    Compute the average hazard summary of the hazard function as a function of the awake bout
    duration. The average hazard represents a summary of the frequency of transitioning from
    an awake to sleep state.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    h_awake : float
        Awake bout average hazard.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate higher frequency in switching from awake to sleep states.

    The average hazard is computed per [1]_:

    .. math::

        h(t_n_i) = \frac{n\left(t_n_i\right)}{n - n^c\left(t_n_{i-1}\right)}
        \har{h} = \frac{1}{m}\sum_{t\in D}h(t)

    where :math:`h(t_n_i)` is the hazard for the awake bout of length :math:`t_n_i`,
    :math:`n(t_n_i)` is the number of bouts of length :math:`t_n_i`, :math:`n` is the total
    number of awake bouts, :math:`n^c(t_n_i)` is the sum number of bouts less than or equal to
    length :math:`t_n_i`, and :math:`t\in D` indicates all bouts up to the maximum length
    (:math:`D`).
    """
    lengths, starts, vals = rle(sleep_predictions)
    wake_lengths = lengths[vals == 0]

    u_al, c_al = unique(wake_lengths, return_counts=True)
    sidx = argsort(u_al)

    c_al = c_al[sidx]
    cs_c_al = insert(cumsum(c_al), 0, 0)

    h_i = c_al / (cs_c_al[-1] - cs_c_al[:-1])

    return sum(h_i) / u_al.size


def sleep_power_law_distribution(sleep_predictions):
    r"""
    Compute the scaling factor for a power law distribution over the sleep bouts lengths.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    alpha : float
        Sleep bout power law distribution scaling parameter.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Larger `alpha` values indicate that the total sleeping time is accumulated with a larger
    portion of shorter sleep bouts.

    The power law scaling factor is computer per [1]_:

    .. math:: 1 + \frac{n_{sleep}}{\sum_{i}\log{t_i / \left(min(t) - 0.5\right)}}

    where :math:`n_{sleep}` is the number of sleep bouts, :math:`t_i` is the duration of the
    :math:`ith` sleep bout, and :math:`min(t)` is the length of the shortest sleep bout.
    """
    lengths, starts, vals = rle(sleep_predictions)
    sleep_lengths = lengths[vals == 1]

    return 1 + sleep_lengths.size / sum(log(sleep_lengths / (sleep_lengths.min() - 0.5)))


def awake_power_law_distribution(sleep_predictions):
    """
    Compute the scaling factor for a power law distribution over the awake bouts lengths.

    Parameters
    ----------
    sleep_predictions : numpy.ndarray
        Boolean array indicating sleep (True = sleeping).

    Returns
    -------
    alpha : float
        Awake bout power law distribution scaling parameter.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are associated with
        mortality in US adults: The NHANES study,” bioRxiv, p. 182337, Aug. 2017,
        doi: 10.1101/182337.

    Notes
    -----
    Larger `alpha` values indicate that the total awake time is accumulated with a larger
    portion of shorter awake bouts.

    The power law scaling factor is computer per [1]_:

    .. math:: 1 + \frac{n_{awake}}{\sum_{i}\log{t_i / \left(min(t) - 0.5\right)}}

    where :math:`n_{awake}` is the number of awake bouts, :math:`t_i` is the duration of the
    :math:`ith` awake bout, and :math:`min(t)` is the length of the shortest awake bout.
    """
    lengths, starts, vals = rle(sleep_predictions)
    wake_lengths = lengths[vals == 0]

    return 1 + wake_lengths.size / sum(log(wake_lengths / (wake_lengths.min() - 0.5)))
