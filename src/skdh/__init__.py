"""
Scikit IMU (:mod:`skdh`)
=========================

.. currentmodule:: skdh

Pipeline Processing
-------------------

.. autosummary::
    :toctree: generated/

    Pipeline
"""
from skdh.version import __version__

from skdh.pipeline import Pipeline

from skdh import utility
from skdh import read
from skdh import preprocessing
from skdh import sleep
from skdh import activity
from skdh import gait
from skdh import sit2stand
from skdh import features

__skdh_version__ = __version__
__all__ = [
    "Pipeline",
    "activity",
    "gait",
    "sit2stand",
    "read",
    "sleep",
    "preprocessing",
    "features",
    "utility",
    "__skdh_version__",
]
