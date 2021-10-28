"""
Binary File Reading (:mod:`skdh.read`)
=======================================

.. currentmodule:: skdh.read

These processes are designed to quickly read in data from various different
wearable devices from their default binary file format.

.. autosummary::
    :toctree: generated/

    ReadCwa
    ReadBin
    ReadGT3X
    ReadApdmH5
    ReadNumpyFile
"""
from skdh.read.axivity import ReadCwa
from skdh.read import axivity
from skdh.read.geneactiv import ReadBin
from skdh.read import geneactiv
from skdh.read.actigraph import ReadGT3X
from skdh.read import actigraph
from skdh.read.apdm import ReadApdmH5
from skdh.read import apdm
from skdh.read.numpy_compressed import ReadNumpyFile
from skdh.read import numpy_compressed
from skdh.read.utility import FileSizeError

__all__ = (
    "ReadCwa",
    "ReadBin",
    "ReadGT3X",
    "ReadApdmH5",
    "ReadNumpyFile",
    "axivity",
    "geneactiv",
    "actigraph",
    "apdm",
    "numpy_compressed",
)
