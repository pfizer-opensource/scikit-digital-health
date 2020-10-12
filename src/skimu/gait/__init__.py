"""
IMU Gait Analysis (:mod:`skimu.gait`)
=====================================

.. currentmodule:: skimu.gait

Pipeline gait processing
------------------------

.. autosummary::
    :toctree: generated/

    Gait

.. _event-level-gait-metrics:

Event Level Gait Metrics
------------------------

.. autosummary::
    :toctree: generated/

    StrideTime
    StanceTime
    SwingTime
    StepTime
    InitialDoubleSupport
    TerminalDoubleSupport
    DoubleSupport
    SingleSupport
    StepLength
    StrideLength
    GaitSpeed
    Cadence
    IntraStepCovarianceV
    IntraStrideCovarianceV
    HarmonicRatioV

.. _bout-level-gait-metrics:

Bout Level Gait Metrics
-----------------------

.. autosummary::
    :toctree: generated/

    PhaseCoordinationIndex
    GaitSymmetryIndex
    StepRegularityV
    StrideRegularityV
    AutocovarianceSymmetryV
    RegularityIndexV
"""
from skimu.gait.gait import Gait
from skimu.gait import gait
from skimu.gait.gait_metrics import *
from skimu.gait import gait_metrics
from skimu.gait import train_classifier
