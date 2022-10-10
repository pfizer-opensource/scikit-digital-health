"""
Utility Functions (:mod:`skdh.utility`)
=======================================

.. currentmodule:: skdh.utility

Binary State Fragmentation Endpoints
------------------------------------

.. autosummary:
    :toctree: generated/

    fragmentation_endpoints.average_duration
    fragmentation_endpoints.state_transition_probability
    fragmentation_endpoints.gini_index
    fragmentation_endpoints.average_hazard
    fragmentation_endpoints.state_power_law_distribution

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

from skdh.utility.fragmentation_endpoints import *
from skdh.utility import fragmentation_endpoints
from skdh.utility.math import *
from skdh.utility import math
from skdh.utility.orientation import correct_accelerometer_orientation
from skdh.utility import orientation
from skdh.utility.windowing import compute_window_samples, get_windowed_view
from skdh.utility import windowing
from skdh.utility import activity_counts
from skdh.utility.activity_counts import *


__all__ = (
    ["math", "windowing", "orientation", "fragmentation_endpoints"]
    + fragmentation_endpoints.__all__
    + math.__all__
    + windowing.__all__
    + orientation.__all__
    + activity_counts.__all__
)
