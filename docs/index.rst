.. _skimu documentation master file, created by
   sphinx-quickstart on Tue Oct  6 14:50:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SciKit-IMU's documentation!
=================================

.. image:: https://github.com/PfizerRD/scikit-imu/workflows/skimu/badge.svg)
   :alt: GitHub Actions Badge

`Scikit-IMU` is a Python package with methods for reading, pre-processing, manipulating, and analyzing Inertial Meausurement Unit data. `SciKit-IMU` contains the following sub-modules:

sit2stand
---------
The `sit2stand` sub-module uses novel algorithms to first detect Sit-to-Stand transitions from lumbar-mounted accelerometer data, and then provide quantitative metrics assessing the performance of the transitions. As gyroscopes impose a significant detriment to battery life due to power consumption, `sit2stand`'s use of acceleration only allows for a single sensor to collect days worth of analyzable data.

gait
----
The `gait` sub-module contains python implementations of existing algorithms from previous literature, as well as a gait classifier for detecting bouts of gait during at-home recordings of data. `gait`'s algorithms detect gait and analyze the gait for metrics such as stride time, stride length, etc from lumbar acceleration data. Adding gyroscope data will also enable several additional features.

read
----
The `read` sub-module contains methods for reading data from several different sensor binary files into python. These have been implemented in low level languages (C, Fortran) to enable significant speed-ups in loading this data over pure-python implementations, or reading csv files

features
--------
The `features` sub-module contains a library of features that can be generated for accelerometer/gyroscope/time-series signals. These features can be used for classification problems, etc. Feature computations are also implemented in Fortran to ensure quicker computation time when dealing with very long (multiple days) recordings.

Requirements
------------
- Python >= 3.6
- numpy >= 1.17.2
- scipy >= 1.3.1
- pandas >= 0.23.4
- lightgbm >= 2.3.0
- pywavelets
- [cython >= 0.29.14] : optional, only if cythonizing .pyx files before building

Contents
--------
.. toctree::
   :maxdepth: 3

   src/installation
   src/testing
   src/usage
   ref/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
