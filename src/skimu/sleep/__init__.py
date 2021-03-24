"""
IMU Sleep Analysis (:mod:`skimu.sleep`)
=======================================

.. currentmodule:: skimu.sleep

Pipeline sleep processing
-------------------------

.. autosummary::
    :toctree: generated/

    Sleep

.. _sleep-metrics:

Sleep Metrics
-------------

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
from skimu.sleep.sleep import Sleep
from skimu.sleep import sleep
from skimu.sleep.endpoints import *
from skimu.sleep import endpoints

__all__ = ["Sleep", "endpoints"] + endpoints.__all__
