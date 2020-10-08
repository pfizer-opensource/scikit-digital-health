"""
IMU Gait Analysis (:mod:`PfyMU.gait`)
====================================

.. currentmodule:: PfyMU.gait

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
from PfyMU.gait.gait import Gait
from PfyMU.gait import gait
from PfyMU.gait.gait_metrics import *
from PfyMU.gait import gait_metrics
from PfyMU.gait import train_classifier
