"""
Sleep-based endpoints

Yiorgos Christakis, Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from abc import ABC, abstractmethod
import logging

from numpy import around, nonzero, diff, argmax, sum, int_, maximum, nan

from skdh.utility import fragmentation_endpoints as frag_endpts

__all__ = [
    "SleepEndpoint",
    "TotalSleepTime",
    "PercentTimeAsleep",
    "NumberWakeBouts",
    "SleepOnsetLatency",
    "WakeAfterSleepOnset",
    "AverageSleepDuration",
    "AverageWakeDuration",
    "SleepWakeTransitionProbability",
    "WakeSleepTransitionProbability",
    "SleepGiniIndex",
    "WakeGiniIndex",
    "SleepAverageHazard",
    "WakeAverageHazard",
    "SleepPowerLawDistribution",
    "WakePowerLawDistribution",
]


class SleepEndpoint(ABC):
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self, name, logname, depends=None):
        """
        Sleep metric base class

        Parameters
        ----------
        name : str
            Name of the metric
        logname : str
            name of an active logger
        depends : {None, list}
            Metric dependencies
        """
        self.name = name
        self.logger = logging.getLogger(logname)

        self._depends = depends

    @abstractmethod
    def predict(self, **kwargs):
        pass


class TotalSleepTime(SleepEndpoint):
    """
    Compute the total time spent asleep from 1 minute epoch sleep predictions.
    """

    def __init__(self):
        super().__init__("total sleep time", __name__)

    def predict(self, sleep_predictions, **kwargs):
        """
        predict(sleep_predictions)

        Parameters
        ----------
        sleep_predictions : numpy.ndarray
            Boolean array indicating sleep (True = sleeping).

        Returns
        -------
        tst : int
            Number of minutes spent asleep during the total sleep opportunity.
        """
        return sum(sleep_predictions)


class PercentTimeAsleep(SleepEndpoint):
    """
    Compute the percent time spent asleep from 1 minute epoch sleep predictions.
    """

    def __init__(self):
        super().__init__("percent time asleep", __name__)

    def predict(self, sleep_predictions, **kwargs):
        """
        predict(sleep_predictions)

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


class NumberWakeBouts(SleepEndpoint):
    """
    Compute the number of waking bouts during the total sleep opportunity, excluding
    the first wake before sleep, and last wake bout after sleep.
    """

    def __init__(self):
        super().__init__("number of wake bouts", __name__)

    def predict(self, sleep_predictions, **kwargs):
        """
        predict(sleep_predictions)

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
        return maximum(
            nonzero(diff(sleep_predictions.astype(int_)) == -1)[0].size - 1, 0
        )


class SleepOnsetLatency(SleepEndpoint):
    """
    Compute the amount of time before the first sleep period in minutes.
    """

    def __init__(self):
        super(SleepOnsetLatency, self).__init__("sleep onset latency", __name__)

    def predict(self, sleep_predictions, **kwargs):
        """
        predict(sleep_predictions)

        Parameters
        ----------
        sleep_predictions : numpy.ndarray
            Boolean array indicating sleep (True = sleeping).

        Returns
        -------
        sol : int
            Total number of minutes spent awake before the first sleep period
        """
        if not sleep_predictions.any():
            return nan  # want it to be undefined if no sleep occurred
        return argmax(sleep_predictions)  # samples = minutes


class WakeAfterSleepOnset(SleepEndpoint):
    """
    Compute the number of minutes awake after the first period of sleep, excluding
    the last wake period after sleep.
    """

    def __init__(self):
        super(WakeAfterSleepOnset, self).__init__("wake after sleep onset", __name__)

    def predict(self, sleep_predictions, **kwargs):
        """
        predict(sleep_predictions)

        Parameters
        ----------
        sleep_predictions : numpy.ndarray
            Boolean array indicating sleep (True = sleeping).

        Returns
        -------
        waso : int
            Total number of minutes spent awake after the first sleep period
        """
        if not sleep_predictions.any():
            return nan  # if never fell asleep then metric should be undefined
        first_epoch, last_epoch = nonzero(sleep_predictions)[0][[0, -1]]
        waso = (last_epoch - first_epoch) - sum(
            sleep_predictions[first_epoch:last_epoch]
        )
        return waso


class AverageSleepDuration(SleepEndpoint):
    r"""
    Compute the average duration of a sleep bout.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate longer bouts of sleep.
    """

    def __init__(self):
        super(AverageSleepDuration, self).__init__("average sleep duration", __name__)

    def predict(self, lengths, starts, values, **kwargs):
        """
        predict(lengths, starts, values)

        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        asp : float
            Average number of minutes per bout of sleep during total sleep opportunity.
        """
        return frag_endpts.average_duration(lengths=lengths, values=values, voi=1)


class AverageWakeDuration(SleepEndpoint):
    r"""
    Compute the average duration of wake bouts during sleep.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate longer bouts of wakefulness.
    """

    def __init__(self):
        super(AverageWakeDuration, self).__init__("average wake duration", __name__)

    def predict(self, lengths, starts, values, **kwargs):
        """
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        awp : float
            Average number of minutes per bout of wake during total sleep opportunity.
        """
        return frag_endpts.average_duration(lengths=lengths, values=values, voi=0)


class SleepWakeTransitionProbability(SleepEndpoint):
    r"""
    Compute the probability of transitioning from sleep state to awake state

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate more frequent switching between states, and as a result
    may indicate greater fragmentation of sleep.

    The implementation is straightforward [1]_, and is simply defined as

    .. math:: satp = \frac{1}{\mu_{sleep}}

    where :math:`\mu_{sleep}` is the mean sleep bout time.
    """

    def __init__(self):
        super(SleepWakeTransitionProbability, self).__init__(
            "sleep wake transition probability", __name__
        )

    def predict(self, lengths, starts, values, **kwargs):
        r"""
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        satp : float
            Sleep to awake transition probability during the total sleep opportunity.
        """
        return frag_endpts.state_transition_probability(
            lengths=lengths, values=values, voi=1
        )


class WakeSleepTransitionProbability(SleepEndpoint):
    r"""
    Compute the probability of transitioning from awake state to sleep state.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate more frequent switching between states, and as a result
    may indicate greater fragmentation of sleep.

    The implementation is straightforward [1]_, and is simply defined as

    .. math:: satp = \frac{1}{\mu_{awake}}

    where :math:`\mu_{awake}` is the mean awake bout time.
    """

    def __init__(self):
        super(WakeSleepTransitionProbability, self).__init__(
            "wake sleep transition probability", __name__
        )

    def predict(self, lengths, starts, values, **kwargs):
        r"""
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        astp : float
            Awake to sleep transition probability during the total sleep opportunity.
        """
        return frag_endpts.state_transition_probability(
            lengths=lengths, values=values, voi=0
        )


class SleepGiniIndex(SleepEndpoint):
    r"""
    Compute the normalized variability of the sleep bouts, also known as the Gini
    Index from economics.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Gini Index values are bounded between 0 and 1, with values near 1 indicating
    the total time accumulating due to a small number of longer bouts, whereas values
    near 0 indicate all bouts contribute more equally to the total time.
    """

    def __init__(self):
        super(SleepGiniIndex, self).__init__("sleep gini index", __name__)

    def predict(self, lengths, starts, values, **kwargs):
        """
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        gini : float
            Sleep normalized variability or Gini Index during total sleep opportunity.
        """
        return frag_endpts.gini_index(lengths=lengths, values=values, voi=1)


class WakeGiniIndex(SleepEndpoint):
    r"""
    Compute the normalized variability of the awake bouts, also known as the Gini
    Index from economics.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Gini Index values are bounded between 0 and 1, with values near 1 indicating
    the total time accumulating due to a small number of longer bouts, whereas values
    near 0 indicate all bouts contribute more equally to the total time.
    """

    def __init__(self):
        super(WakeGiniIndex, self).__init__("wake gini index", __name__)

    def predict(self, lengths, starts, values, **kwargs):
        """
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        gini : float
            Awake normalized variability or Gini Index during total sleep opportunity.
        """
        return frag_endpts.gini_index(lengths=lengths, values=values, voi=0)


class SleepAverageHazard(SleepEndpoint):
    r"""
    Compute the average hazard summary of the hazard function as a function of the
    sleep bout duration. The average hazard represents a summary of the frequency
    of transitioning from a sleep to awake state.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

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

    def __init__(self):
        super(SleepAverageHazard, self).__init__("sleep average hazard", __name__)

    def predict(self, lengths, starts, values, **kwargs):
        r"""
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        h_sleep : float
            Sleep bout average hazard.
        """
        return frag_endpts.average_hazard(lengths=lengths, values=values, voi=1)


class WakeAverageHazard(SleepEndpoint):
    r"""
    Compute the average hazard summary of the hazard function as a function of the
    awake bout duration. The average hazard represents a summary of the frequency
    of transitioning from an awake to sleep state.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Higher values indicate higher frequency in switching from awake to sleep states.

    The average hazard is computed per [1]_:

    .. math::

        h(t_n_i) = \frac{n\left(t_n_i\right)}{n - n^c\left(t_n_{i-1}\right)}
        \har{h} = \frac{1}{m}\sum_{t\in D}h(t)

    where :math:`h(t_n_i)` is the hazard for the awake bout of length :math:`t_n_i`,
    :math:`n(t_n_i)` is the number of bouts of length :math:`t_n_i`, :math:`n` is
    the total number of awake bouts, :math:`n^c(t_n_i)` is the sum number of bouts
    less than or equal to length :math:`t_n_i`, and :math:`t\in D` indicates all
    bouts up to the maximum length (:math:`D`).
    """

    def __init__(self):
        super(WakeAverageHazard, self).__init__("wake average hazard", __name__)

    def predict(self, lengths, starts, values, **kwargs):
        r"""
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        h_awake : float
            Awake bout average hazard.
        """
        return frag_endpts.average_hazard(lengths=lengths, values=values, voi=0)


class SleepPowerLawDistribution(SleepEndpoint):
    r"""
    Compute the scaling factor for a power law distribution over the sleep bouts
    lengths.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

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

    def __init__(self):
        super(SleepPowerLawDistribution, self).__init__(
            "sleep power law distribution", __name__
        )

    def predict(self, lengths, starts, values, **kwargs):
        r"""
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        alpha : float
            Sleep bout power law distribution scaling parameter.
        """
        return frag_endpts.state_power_law_distribution(
            lengths=lengths, values=values, voi=1
        )


class WakePowerLawDistribution(SleepEndpoint):
    r"""
    Compute the scaling factor for a power law distribution over the awake bouts
    lengths.

    References
    ----------
    .. [1] J. Di et al., “Patterns of sedentary and active time accumulation are
        associated with mortality in US adults: The NHANES study,” bioRxiv, p. 182337,
        Aug. 2017, doi: 10.1101/182337.

    Notes
    -----
    Larger `alpha` values indicate that the total awake time is accumulated with
    a larger portion of shorter awake bouts.

    The power law scaling factor is computer per [1]_:

    .. math:: 1 + \frac{n_{awake}}{\sum_{i}\log{t_i / \left(min(t) - 0.5\right)}}

    where :math:`n_{awake}` is the number of awake bouts, :math:`t_i` is the duration
    of the :math:`ith` awake bout, and :math:`min(t)` is the length of the shortest
    awake bout.
    """

    def __init__(self):
        super(WakePowerLawDistribution, self).__init__(
            "wake power law distribution", __name__
        )

    def predict(self, lengths, starts, values, **kwargs):
        r"""
        predict(lengths, starts, values)
        Parameters
        ----------
        lengths : numpy.ndarray
            Lengths of bouts.
        starts : numpy.ndarray
            Indices of bout starts
        values : numpy.ndarray
            Value of the bout.

        Returns
        -------
        alpha : float
            Awake bout power law distribution scaling parameter.
        """
        return frag_endpts.state_power_law_distribution(
            lengths=lengths, values=values, voi=0
        )
