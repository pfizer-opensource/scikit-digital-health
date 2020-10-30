"""
Scikit IMU (:mod:`skimu`)
=======================================

.. currentmodule:: skimu

Utility Functions
-----------------

.. autosummary::
    :toctree: generated/

    utility.compute_window_samples
    utility.get_windowed_view
"""
from skimu.version import __version__

from skimu.pipeline import Pipeline
from skimu import utility

from skimu import gait
from skimu import sit2stand
from skimu import features

__skimu_version__ = __version__
__all__ = ['Pipeline', 'gait', 'sit2stand', 'features', 'utility', '__skimu_version__']
