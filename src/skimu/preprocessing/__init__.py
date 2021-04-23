"""
Inertial Data Preprocessing (:mod:`skimu.preprocessing`)
=======================================================

.. currentmodule:: skimu.preprocessing

Sensor Calibration
------------------

.. autosummary::
    :toctree: generated/

    CalibrateAccelerometer
"""
from skimu.preprocessing.calibrate import CalibrateAccelerometer
from skimu.preprocessing import calibrate
from skimu.preprocessing.wear_detection import DetectWear
from skimu.preprocessing import wear_detection

__all__ = ("CalibrateAccelerometer", "calibrate", "DetectWear", "wear_detection")
