"""
Signal Features (:mod:`skdh.features`)
======================================

.. currentmodule:: skdh.features

Combined Feature Processing
---------------------------

.. autosummary::
    :toctree: generated/

    Bank

Signal Features
---------------

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
    RangePowerSum
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

from skdh.features.core import *
from skdh.features import core
from skdh.features.lib import *
from skdh.features import lib

__all__ = core.__all__ + lib.__all__
