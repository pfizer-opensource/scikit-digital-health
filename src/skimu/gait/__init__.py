"""
IMU Gait Analysis (:mod:`skimu.gait`)
=====================================

.. currentmodule:: skimu.gait

Pipeline gait processing
------------------------

.. autosummary::
    :toctree: generated/

    Gait

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

Bout Level Gait Metrics
-----------------------

.. autosummary::
    :toctree: generated/

    PhaseCoordinationIndex
    GaitSymmetryIndex
    StepRegularityV
    StrideRegularityV
    AutocovarianceSymmetryV
"""
from skimu.gait.gait import Gait
from skimu.gait import gait
from skimu.gait.gait_metrics import *
from skimu.gait import gait_metrics
from skimu.gait import train_classifier
