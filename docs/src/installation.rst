Installation
============

::

    pip install git+ssh://git@github.com/PfizerRD/PfyMU.git

If you want to install from a different branch (ie development):

::

    pip install git+ssh://git@github.com/PfizerRD/PfyMU@development


Finally, PfyMU is distributed with the `.c` files for cython extensions, however, if you would like to build these `.c` files from the `.pyx` cython files, simply set an environment variable `CYTHONIZE=True`:

::

    export CYTHONIZE=True
    pip install git+ssh://git@github.com/PfizerRD/PfyMU.git
