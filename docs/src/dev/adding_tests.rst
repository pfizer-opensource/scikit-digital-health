.. _adding-tests:

############
Adding tests
############

Tests are required for any contributions, and should have 100% statement coverage.

Tests are written with `PyTest <https://docs.pytest.org/en/stable/>`_, and coverage is tested 
with `coverage <https://coverage.readthedocs.io/en/coverage-5.3/>`_. These and other testing 
requirements are listed in the optional ``test`` dependencies in ``pyproject.toml``.

Adding tests for a new module
-----------------------------

In order to make sure that any tests are run on installed versions of ``scikit-digital-health``, the test directory is outside the ``src`` directory. Again, convenient base process testing classes are available to make setting up the basic testing easy and quick.

1. Create a new directory for your new module under the ``test/`` directory.

    * Add an empty ``__init__.py`` file to allow for relative importing.
    * Add the test file, eg ``test_preprocessing.py`` for this example.
    * Add a ``conftest.py`` file if desired::

        scikit-digital-health
        ├── src
        ├── test
        │   └──preprocessing
        │       ├── __init__.py
        │       ├── conftest.py
        │       └── test_preprocessing.py

2. Inside ``test_preprocessing.py``, import the base process testing class, set a few options, and the basic tests will be completed (when provided with sample and truth data!):

.. code:: python

    # test/preprocessing/test_preprocessing.py
    import pytest

    # import BaseProcessTester, and resolve_data_path - a useful utility for make sure 
    # tests run in multiple locations
    from ..base_conftest import *

    from skdh.preprocessing import Preprocessing

    class TestPreprocessing(BaseProcessTester):
        @classmethod
        def setup_class(cls):
            super().setup_class()  # make sure to call the super method

            # override specific necessary attributes
            """
            resolve_data_path() takes 2 arguments:
            1. the file name
            2. the module name (ie the folder name this script is in)
            """
            cls.sample_data_file = resolve_data_path('preprocess_data.h5', 'preprocessing')
            cls.truth_data_file = resolve_data_path('preprocess_data.h5', 'preprocessing')
            cls.truth_suffix = None  # if not none, means that the truth data is in the path "Truth/{truth_suffix}" in the truth h5 file
            cls.truth_data_keys = [  # list of keys to check against truth data
                'accel'
            ]

            cls.process = Preprocessing(attr1=5, attr2=10.0)
        
        """
        Adding additional tests, for errors, edge cases, anything that can't be 
        accomplished with the provided default structure can be easily added by
        defining new functions under this class.  

        Note that if the default .test method is not working for your case, just overwrite it.
        """
        def test_error_attr1_string(self):
            with pytest.raises(ValueError):
                Preprocessing(attr1=5, attr2='string')

* The above file would be all that is needed to test that the output `accel` matches the data contained in the truth file.

Creating the sample/truth data files
------------------------------------

* The sample and truth data files are h5 files, with the below formats/keys:

.. code::

    sample.h5  # can have one or more of any of the below keys
    ├── time
    ├── accel
    ├── gyro
    └── temperature

* The ``BaseProcessTester`` class will automatically look for these keys. If you need to specify more/other keys, the list is stored in ``cls.sample_data_keys``:

.. code:: python

    @classmethod
    def setup_class(cls):
        ...
        # add more keys to look for
        cls.sample_data_keys.extend([
            'extra_key1',
            'extra_key2'
        ])
        ...

* The truth data is very similar. For the ``preprocessing`` example, testing ``accel`` the h5 file would look like this:

.. code::

    truth.h5
    ├── Truth
    │   └── accel

* Alternatively, if the ``cls.truth_suffix`` is set to somethign else, (eg ``preproc``) then the structure would be as follows:

.. code::

    truth.h5
    ├── Truth
    │   └── preproc
    │       └── accel

* Finally, the sample and truth data can be in 1 file:

.. code::

    sample_truth.h5
    ├── Truth
    │   └── accel
    ├── time
    ├── accel
    ├── gyro
    └── temperature