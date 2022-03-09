"""
Scikit Digital Health (:mod:`skdh`)
===================================

.. currentmodule:: skdh

Pipeline Processing
-------------------

.. autosummary::
    :toctree: generated/

    Pipeline
"""
# from skdh.version import __version__, __minimum_version__

from skdh.pipeline import Pipeline
from skdh.base import BaseProcess

from skdh import utility
from skdh import io
from skdh import preprocessing
from skdh import sleep
from skdh import activity
from skdh import gait
from skdh import sit2stand
from skdh import features


# get the version number from the file
with open("VERSION", "r") as f:
    __version__ = f.readline().strip()
    __skdh_version__ = __version__

# ==============================================================================
# MINIMUM VERSION FOR GUARANTEED BACKWARDS COMPATIBILITY
# ==============================================================================
# set the minimum version
__minimum_version__ = "0.9.10"

__all__ = [
    "Pipeline",
    "BaseProcess",
    "activity",
    "gait",
    "sit2stand",
    "io",
    "sleep",
    "preprocessing",
    "features",
    "utility",
    "__skdh_version__",
]
