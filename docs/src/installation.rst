Installation
============

Currently (March 2022), only source distributions are available. The plan is to
provide packages on both PyPI and conda-forge, which will ideally have the necessary
libraries provided with the wheel/installation. However, for now, all the requirements
for building from source must be met.

.. tabbed:: conda

    ::

        conda install -c conda-forge scikit-digital-health

.. tabbed:: pip
    :selected:

    ::

        pip install scikit-digital-health

.. tabbed:: pip from GitHub

    ::

        pip install git+ssh://git@github.com/PfizerRD/scikit-digital-health.git
        # install from a specific tag
        pip install git+ssh://git@github.com/PfizerRD/scikit-digital-health@0.9.10
        # install from a specific branch
        pip install git+ssh://git@github.com/PfizerRD/scikit-digital-health@development

Run-time requirements
^^^^^^^^^^^^^^^^^^^^^

- numpy >=1.17.2
- scipy >=1.3.1
- pandas >=1.0.0
- lightgbm >=2.3.0
- pywavelets
- scikit-learn
- h5py
- matplotlib
- packaging
- pyyaml
- importlib_resources [Python < 3.7]

Building from Source
====================
In order to build from source, the following are required:

- C compiler (currently has only been tested with GCC)
- Fortran compiler (currently only been tested with GFortran)
- `libzip <https://libzip.org/>`_
- setuptools
- numpy >=1.17.2

If you are using Conda, libzip can easily be installed via::

    conda install -c conda-forge libzip

C/Fortran Compilers
^^^^^^^^^^^^^^^^^^^

NumPy only supports specific compilers on each platform. To get a list of available
compilers, when in the `scikit-digital-health` directory, running::

    # Get a list of platform supported/installed C compilers
    python setup.py build_ext --help-compiler
    # Get a list of platform supported/installed Fortran compilers
    python setup.py build_ext --help-fcompiler


Testing the Build
^^^^^^^^^^^^^^^^^

In order to test a build, you will need to clone or download the source code from
`GitHub <https://github.com/PfizerRD/scikit-digital-health>`_, as the tests are isolated
from access to the package source code. Additionally, the following are requirements
to run the test suite:

- pytest
- psutil
- tables
- coverage [optional, for running test coverage]

From the top level `scikit-digital-health` directory, run::

    pytest test/

The output should look similar to::

    ============================================================================
    platform darwin -- Python 3.10.2, pytest-7.0.1, pluggy-1.0.0
    rootdir: /Users/lukasadamowicz/Documents/Packages/scikit-digital-health
    collected 434 items

    test/activity/test_activity_core.py .....                             [  1%]
    test/activity/test_activity_endpoints.py ................             [  4%]
    # ... rest of test results ...

    =========================== warnings summary ===============================
    # ... warnings content ...
    ========= 427 passed, 7 skipped, 61 warnings in 6.30s ======================

