"""
Signal Features (:mod:`PfyMU.features`)
====================================

.. currentmodule:: PfyMU.features

Combined Feature Processing
---------------------------

.. autosummary::
    :toctree: generated/

    Bank

Features
--------

.. autosummary::
    :toctree: generated/

    Mean
    MeanCrossRate
    StdDev
    Skewness
    Kurtosis
    Range
    IQR
    RMS
    Autocorrelation
    LinearSlope
    SignalEntropy
    SampleEntropy
    PermutationEntropy
    DominantFrequency
    DominantFrequencyValue
    PowerSpectralSum
    SpectralFlatness
    SpectralEntropy
    ComplexityInvariantDistance
    RangeCountPercentage
    RatioBeyondRSigma
    JerkMetric
    DimensionlessJerk
    SPARC
    DetailPower
    DetailPowerRatio

Utility Functions
-----------------------

.. autosummary::
    :toctree: generated/

    compute_window_samples
    get_windowed_view
"""
from PfyMU.features.core import *
from PfyMU.features import core
from PfyMU.features.utility import *
from PfyMU.features import utility
from PfyMU.features.lib import *
from PfyMU.features import lib

__all__ = core.__all__ + utility.__all__ + lib.__all__
