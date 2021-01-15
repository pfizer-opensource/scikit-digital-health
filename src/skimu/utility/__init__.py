"""
Utility Functions (:mod:`skimu.utility`)
================================

.. currentmodule:: skimu.utility

Utility Functions
-----------------

.. autosummary::
    :toctree: generated/

    windowing.compute_window_samples
    windowing.get_windowed_view
"""

from skimu.utility.windowing import compute_window_samples, get_windowed_view
from skimu.utility import windowing


__all__ = ["windowing"] + windowing.__all__
