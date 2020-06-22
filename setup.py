"""
setup.py file for installation of a python package.
A general guideline with options for the setuptools.setup can be found here:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""
class SetupError(Exception):
    pass

from setuptools import setup, find_packages, Extension
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ModuleNotFoundError:
    USE_CYTHON = False

# try:
#     from numpy.distutils.core import Extension as npExtension
# except ModuleNotFoundError:
#     raise SetupError('numpy must be installed to compile fortran modules')

with open("README.md", "r") as fh:
    long_description = fh.read()

# these lines allow 1 file to control the version, so only 1 file needs to be updated per version change
fid = open("PfyMU/version.py")
vers = fid.readlines()[-1].split()[-1].strip("\"'")
fid.close()

if USE_CYTHON:
    extensions = cythonize(Extension('PfyMU.features.lib._cython.*',
                                     sources=['PfyMU/features/lib/_cython/*.pyx'], libraries=['m']),
                           compiler_directives={'language_level': 3})
else:
    extensions = [Extension('PfyMU.features.lib._cython.*', sources=['PfyMU/features/lib/_cython/*.c'])]

setup(
    name="PfyMU",
    version=vers,
    author="Pfizer DMTI Analytics",
    author_email="",
    description="Python general purpose IMU analysis and processing package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",  # project url, most likely a github link
    # download_url="https://pypi.org/signal_features",  # link to where the package can be downloaded, most likely PyPI
    # project_urls={"Documentation": "https://signal_features.readthedocs.io/en/latest/"},
    include_package_data=True,  # set to True if you have data to package, ie models or similar
    # package_data={'package': ['*.csv']},  # currently adds any csv files alongside the top level __init__.py
    package_data={'PfyMU.tests.data': ['*.h5'],
                  'PfyMU.features.lib._cython': ['*.c', '*.pxd']},
    ext_modules=extensions,
    packages=find_packages(),  # automatically find required packages
    license="MIT",
    python_requires=">=3.6",  # Version of python required
    install_requires=[
        'cython>=0.29.14',
        'scipy>=1.3.1',
        'statsmodels>=0.10.1',
        'setuptools>=41.4.0',
        'pandas>=0.23.4',
        'numpy>=1.17.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7",
    ],
)