"""
Signal Features (:mod:`skimu.features`)
=======================================

.. currentmodule:: skimu.features

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
"""
from skimu.features.core import *
from skimu.features import core
from skimu.features.lib import *
from skimu.features import lib

__all__ = core.__all__ + utility.__all__ + lib.__all__
