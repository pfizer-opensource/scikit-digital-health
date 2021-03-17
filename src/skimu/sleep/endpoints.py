"""
Sleep-based endpoints

Yiorgos Christakis, Lukas Adamowicz
Pfizer DMTI 2019-2021
"""
from numpy import around, nonzero, diff, argmax, sum, int_


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
