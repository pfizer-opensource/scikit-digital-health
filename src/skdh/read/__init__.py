"""
Binary File Reading (:mod:`skdh.read`)
=======================================

.. currentmodule:: skdh.read

Binary File Readers
-------------------

.. autosummary::
    :toctree: generated/

    ReadCWA
    ReadBin
    ReadGT3X
    ReadApdmH5
"""
from skdh.read.axivity import ReadCwa
from skdh.read import axivity
from skdh.read.geneactiv import ReadBin
from skdh.read import geneactiv
from skdh.read.actigraph import ReadGT3X
from skdh.read import actigraph
from skdh.read.apdm import ReadApdmH5
from skdh.read import apdm

__all__ = (
    "ReadCwa",
    "ReadBin",
    "ReadGT3X",
    "ReadApdmH5",
    "axivity",
    "geneactiv",
    "actigraph",
    "apdm",
)


class FileSizeError(Exception):
    pass

