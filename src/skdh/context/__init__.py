"""
Context Detection (:mod:`skdh.context`)
=======================================

.. currentmodule:: skdh.context

Context Detection
-----------------
This module contains various methods to detect context from inertial sensor data.
Examples of context detection could include the detection of gait bouts - clean bouts
of walking from which gait metrics can be computed, ambulation bouts - periods of
walking-similar activities from which step counts can be estimated, motion detection
- various methods for filtering out periods of non-movement, and others.

.. autosummary::
    :toctree: generated/

    Ambulation
    PredictGaitLumbarLgbm

"""

from skdh.context.ambulation_classification import Ambulation
from skdh.context.motion_detect import MotionDetectRCoV, MotionDetectRCoVMaxSD
from skdh.context.gait_classification import PredictGaitLumbarLgbm
