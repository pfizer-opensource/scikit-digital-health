"""
Inertial Data Preprocessing (:mod:`skdh.preprocessing`)
=======================================================

.. currentmodule:: skdh.preprocessing

Day Windowing
-------------

.. autosummary::
    :toctree: generated/

    GetDayWindowIndices

Sensor Calibration
------------------

.. autosummary::
    :toctree: generated/

    CalibrateAccelerometer

Wear Detection
--------------

.. autosummary::
    :toctree: generated/

    DETACH
    CountWearDetection
    CtaWearDetection
    AccelThresholdWearDetection
"""

from skdh.preprocessing.window_days import GetDayWindowIndices
from skdh.preprocessing import window_days
from skdh.preprocessing.calibrate import CalibrateAccelerometer
from skdh.preprocessing import calibrate
from skdh.preprocessing.wear_detection import (
    AccelThresholdWearDetection,
    CtaWearDetection,
    DETACH,
    CountWearDetection,
)
from skdh.preprocessing import wear_detection

__all__ = (
    "GetDayWindowIndices",
    "window_days",
    "CalibrateAccelerometer",
    "calibrate",
    "AccelThresholdWearDetection",
    "CtaWearDetection",
    "DETACH",
    "CountWearDetection",
    "wear_detection",
)
