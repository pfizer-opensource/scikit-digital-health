"""
Inertial Data Preprocessing (:mod:`skdh.preprocessing`)
=======================================================

.. currentmodule:: skdh.preprocessing

Sensor Calibration
------------------

.. autosummary::
    :toctree: generated/

    CalibrateAccelerometer

Wear Detection
--------------

.. autosummary::
    :toctree: generated/

    DetectWear
"""
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
    "CalibrateAccelerometer",
    "calibrate",
    "AccelThresholdWearDetection",
    "CtaWearDetection",
    "DETACH",
    "CountWearDetection",
    "wear_detection",
)
