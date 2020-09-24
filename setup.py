import os
import sys
import textwrap
import warnings
from setuptools import setup, find_packages


CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU GPL v3
Programming Language :: Fortran 95
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: MacOS
"""


REQUIREMENTS = [
    'cython>=0.29.14',
    'numpy>=1.17.2',
    'scipy>=1.3.1',
    'pandas>=0.23.4',
    'lightgbm>=2.3.0'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

# these lines allow 1 file to control the version, so only 1 file needs to be updated per version change
fid = open("src/PfyMU/version.py")
vers = fid.readlines()[-1].split()[-1].strip("\"'")
fid.close()

setup(
    name="PfyMU",
    version=vers,
    author="Pfizer DMTI Analytics",
    author_email="",
    description="Python general purpose IMU analysis and processing package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PfizerRD/PfyMU",  # project url, most likely a github link
    # download_url="https://pypi.org/signal_features",  # link to where the package can be downloaded, most likely PyPI
    # project_urls={"Documentation": "https://signal_features.readthedocs.io/en/latest/"},
    include_package_data=True,  # set to True if you have data to package, ie models or similar
    # package_data={'package': ['*.csv']},  # currently adds any csv files alongside the top level __init__.py
    # package_data={'PfyMU.tests.data': ['*.h5'],
    #               'PfyMU.features.lib._cython': ['*.c', '*.pxd']},
    packages=find_packages('src'),  # automatically find sub-packages
    package_dir={'': 'src'},
    license="MIT",
    python_requires=">=3.6",  # Version of python required
    install_requires=REQUIREMENTS,
    classifiers=CLASSIFIERS,
)

