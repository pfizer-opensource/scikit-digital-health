"""
Scikit IMU (:mod:`skimu`)
=========================

.. currentmodule:: skimu

Pipeline Processing
-------------------

.. autosummary::
    :toctree: generated/

    Pipeline
"""
from skimu.version import __version__

from skimu.pipeline import Pipeline

from skimu import utility
from skimu import read
from skimu import preprocessing
from skimu import sleep
from skimu import gait
from skimu import sit2stand
from skimu import features

__skimu_version__ = __version__
__all__ = [
    "Pipeline",
    "gait",
    "sit2stand",
    "read",
    "sleep",
    "preprocessing",
    "features",
    "utility",
    "__skimu_version__"
]
