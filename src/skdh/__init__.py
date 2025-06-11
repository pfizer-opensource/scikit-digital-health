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

from sys import version_info

if version_info >= (3, 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("scikit-digital-health")
else:  # pragma: no cover
    import importlib_metadata

    __version__ = importlib_metadata.version("scikit-digital-health")

__minimum_version__ = "0.9.10"

from skdh.pipeline import Pipeline
from skdh.base import BaseProcess, handle_process_returns

from skdh import utility
from skdh import io
from skdh import preprocessing
from skdh import sleep
from skdh import activity
from skdh import gait_old
from skdh import gait
from skdh import sit2stand
from skdh import features
from skdh import context
from skdh import completeness

__skdh_version__ = __version__


__all__ = [
    "Pipeline",
    "BaseProcess",
    "activity",
    "gait_old",
    "gait",
    "sit2stand",
    "io",
    "sleep",
    "preprocessing",
    "features",
    "utility",
    "context",
    "completeness",
    "__skdh_version__",
]
