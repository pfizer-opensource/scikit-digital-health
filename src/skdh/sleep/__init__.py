"""
IMU Sleep Analysis (:mod:`skdh.sleep`)
======================================

.. currentmodule:: skdh.sleep

Pipeline sleep processing
-------------------------

.. autosummary::
    :toctree: generated/

    Sleep

.. _sleep-metrics:

Sleep Endpoints
---------------

.. autosummary::
    :toctree: generated/

    TotalSleepTime
    PercentTimeAsleep
    NumberWakeBouts
    SleepOnsetLatency
    WakeAfterSleepOnset
    AverageSleepDuration
    AverageWakeDuration
    SleepWakeTransitionProbability
    WakeSleepTransitionProbability
    SleepGiniIndex
    WakeGiniIndex
    SleepAverageHazard
    WakeAverageHazard
    SleepPowerLawDistribution
    WakePowerLawDistribution


Background Information
----------------------

TODO


Adding Custom Sleep Metrics
---------------------------

TODO

"""
from skdh.sleep.sleep import Sleep
from skdh.sleep import sleep
from skdh.sleep.endpoints import *
from skdh.sleep import endpoints

__all__ = ["Sleep", "endpoints"] + endpoints.__all__
