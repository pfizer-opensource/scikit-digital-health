"""
Binary File Reading (:mod:`skimu.read`)
=======================================

.. currentmodule:: skimu.read

Binary File Readers
-------------------

.. autosummary::
    :toctree: generated/

    ReadCWA
    ReadBin
    ReadGT3X
"""
from skimu.read.axivity import ReadCWA
from skimu.read import axivity
from skimu.read.geneactiv import ReadBin
from skimu.read import geneactiv
from skimu.read.actigraph import ReadGT3X
from skimu.read import actigraph

__all__ = ('ReadCWA', 'ReadBin', 'ReadGT3X', 'axivity', 'geneactiv', 'actigraph')
