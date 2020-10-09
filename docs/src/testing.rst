Testing
=======

Testing has additional requirements:

- pytest
- h5py

All tests are automatically run each time a pull request is issued into the master branch. If for some reason you want to run the tests locally, first clone the repository (tests are no longer distributed with the package itself, saving on install/download size). Once the repository has been downloaded, make sure `scikit-imu` is installed, navigate to the local directory of `scikit-imu`, and simply run `pytest`:

::

    # clone the repository
    git clone ssh+git@github.com:PfizerRD/scikit-imu.git
    # move into the scikit-imu directory
    cd scikit-imu
    # make sure scikit-imu is installed - this is necessary if you haven't already installed it through pip (see above)
    pip install .
    # run tests with higher verbosity
    pytest -v