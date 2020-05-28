"""
setup.py file for installation of a python package.
A general guideline with options for the setuptools.setup can be found here:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = cythonize(
    [
        Extension(
          '*',
          sources = ['*.pyx'],
          libraries = ['m'],
          include_dirs = [np.get_include()]
        )
    ],
    compiler_directives = {'language_level': 3}
)

setup(ext_modules=extensions)
