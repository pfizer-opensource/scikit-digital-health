"""
Utility Functions (:mod:`skimu.utility`)
================================

.. currentmodule:: skimu.utility

Misc. Math Functions
--------------------

.. autosummary::
    :toctree: generated/

    math.rolling_mean
    math.rolling_sd
    math.rolling_skewness
    math.rolling_kurtosis


Windowing Functions
-------------------

.. autosummary::
    :toctree: generated/

    windowing.compute_window_samples
    windowing.get_windowed_view
"""

from skimu.utility.math import rolling_mean, rolling_sd, rolling_skewness, rolling_kurtosis
from skimu.utility import math
from skimu.utility.windowing import compute_window_samples, get_windowed_view
from skimu.utility import windowing


__all__ = ["math", "windowing"] + math.__all__ + windowing.__all__
