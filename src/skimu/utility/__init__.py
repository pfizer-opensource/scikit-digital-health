"""
Utility Functions (:mod:`skimu.utility`)
================================

.. currentmodule:: skimu.utility

Misc. Math Functions
--------------------

.. autosummary::
    :toctree: generated/

    math.moving_mean
    math.moving_sd
    math.moving_skewness
    math.moving_kurtosis
    math.moving_median

Orientation Functions
---------------------

.. autosummary::
    :toctree: generated/

    orientation.correct_accelerometer_orientation


Windowing Functions
-------------------

.. autosummary::
    :toctree: generated/

    windowing.compute_window_samples
    windowing.get_windowed_view
"""

from skimu.utility.math import *
from skimu.utility import math
from skimu.utility.orientation import correct_accelerometer_orientation
from skimu.utility import orientation
from skimu.utility.windowing import compute_window_samples, get_windowed_view
from skimu.utility import windowing


__all__ = (
    ["math", "windowing", "orientation"]
    + math.__all__
    + windowing.__all__
    + orientation.__all__
)
