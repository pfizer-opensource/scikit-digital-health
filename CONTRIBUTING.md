# Coding Standards
Everything inside the `/src` directory should follow PEP8 standards, and pass tests run by `flake8`, which is checked as part of the CI process on any pull requests into the `master` branch. There are 2 exceptions which are not checked (specified in `.flake8`): 
1. in `__init__.py` files, * imports may be used, and imports may be unused
2. Line length is set to a maximum of 99 (Python standard is 79, which is a bit short)

In order to make sure these standards are met, __before__ issuing a pull request, please check that your code conforms to these standards through the following:

1. Make sure flake8 is installed - `conda install flake8` or `pip install flake8` (note that the python version here is important - flake8 will test against syntax for the version it is installed alongside)
2. Navigate to the root directory and run `flake8 src/`
3. Make sure there are no warnings. If there are no warnings, the code is set to be in a pull request

# Automatic Testing
Through GitHub Actions, the test suite (using `pytest`) will be run through on different OSs and python versions. To make sure that your tests are working properly (at least on your OS/python version), at the very least you should run `pytest` from the root directory before any pull requests

# Adding modules
The goal of `PfyMU` is to have one package with a defined architechture that allows for easy pipeline generation with multiple stages that may or may not depend on previous stages. To that end, there are several pre-defined base classes that will help setting up modules that are intended to directly interface with the pipeline infrastructure.

### 1. Create a new module directory
Under `src/PfyMU/` create a new directory with the desired name (for this example, we will use `preprocessing`), and create the normal files for a new python package (`__init__.py`, etc)

### 2. Create the module class that will be added to the pipeline
Below is an example file that contains the class that will be added to the pipeline to use its processing

```python
# src/PfyMU/preprocessing/preprocessing.py
import ...  # import installed modules (eg numpy, etc)

from PfyMU.base import _BaseProcess  # import the base process class

class PreProcessing(_BaseProcess):
    def __init__(self, attr1=None, attr2=None):
        """
        Class to implement any preprocessing steps

        ...
        """
        # make sure to call the super method
        super().__init__('Preprocessing')  # check for the required parameters, in this case it is the human-readable name of this process

        self.attr1 = attr1
        ...
    
    """
    This is the function that will actually run in the pipeline. It needs to have the above call - _predict(self, *, **kwargs). The "*" after self means that arguments must be passed in as key-word arguments. If you need specific names for arguments (e.g. time and accel in this case), put in the function declaration. **kwargs must come last, as additional arguments may be passed in within the pipeline architecture.

    Complete the documentation for this function using the numpydoc format, documenting the arguments that are needed by the function. DO NOT document **kwargs

    NOTE the units for time are [seconds since 1970/0/0 00:00:00], acceleration [g], angular velocity [deg/s]
    """
    def _predict(self, *, time=None, accel=None, **kwargs):
        """
        Do the preprocessing step

        Paramters
        ---------
        time : numpy.ndarray
            (N, ) array of timestamps in unix time, in seconds (seconds since 1970/0/0 0:00:00)
        accel : numpy.ndarray
            (N, 3) array of acceleration values, with units of 'g'
        """
        # call the super method
        super()._predict(time=time, accel=accel, **kwargs)  # pass in the key-word arguments, as well as the key-word argument dictionary

        # do any preprocessing - this can either be functions that are referenced here, or just all inside _predict
        accel1 = step1(accel)
        accel2 = step2(accel)

        # If you need something passed in thats not a default/standard argument (ie it might come through kwargs), use the following:
        if necessary_item in kwargs:
            nec_item = kwargs[necessary_item]
        else:
            raise KeyError(f'{necessary_item} not in the additional arguments passed into predict')

        """
        FINALLY: the return of the _predict function needs to be 2 items
        1. a dictionary of the input to _predict, and anything that might be needed in other stages
        2. anything else that needs to be returned, results, etc
        """
        # for this specific case, the goal is to modify the acceleration (hence preprocessing, the modified version needs to be returned in the input dictionary)
        # additionally, there are no specific results from this step, so None is returned as the second argument
        # the neatest way to return the inputs, plus anything declared in the function declaration is to update the kwargs variable, and return it
        kwargs.update({self._time: time, self._acc: accel2})
        """
        Note that the _BaseProcess class has several attributes which help keep track of the names of time, accel, etc, which should help minimize
        work if these names ever change. However, they can't be readily used in function declarations.
        """
        return kwargs, None
```

### 3. Make sure everything is setup/imported
Make sure all imports are handled in `src/PfyMU/preprocessing/__init__.py`, as well as adding `preprocessing` imports to the `src/PfyMU/__init__.py`.

### 4. Make any additions to setup.py
If you don't have any data files (any non python files that need to be distributed with the package), or low level (c, cython, or fortran) extensions, everything should be good for the actual module, and you can skip to the Testing section

If you do have data files or extensions, do the following in `setup.py` (the main one in the root directory)

#### 4a. Data Files
For data files, find the `def configuration` function in `setup.py`, and locate the DATA FILES section, and add any data files that you have. If you have a lot of files in one directory, you can add a whole directory (but be careful, random files such as caches will be included as well):

```python
# setup.py
...

def configuration(parent_package='', top_path=None):
    ...
    # DATA FILES
    # ========================
    config.add_data_files([
        'src/PfyMU/gait/model/final_features.json',
        'src/PfyMU/gait/model/lgbm_gait_classifier_no-stairs.lgbm',
        'src/PfyMU/preprocessing/data/preprocessing_info.dat'        # Added this file
    ])

    # alternatively add this directory, any files/folders under this directory will be added recursively
    config.add_data_dir('src/PfyMU/preprocessing/data')
    # ========================

    config.get_version('src/PfyMU/version.py')

    return config
```

#### 4b. Extensions
For extensions, again in the `def configuration` function in `setup.py`, locate the EXTENSIONS section, and add your extensions:

```python
# setup.py
...
def configuration(parent_package='', top_path=None):
    ...
    # EXTENSIONS
    # ========================
    # Fortran code that is NOT being compiled with f2py - it is being built as a fortran function that will be imported into C code
    config.add_library('fcwa_convert', sources='src/PfyMU/read/_extensions/cwa_convert.f95')
    # C code that contains the necessary CPython API calls to allow it to be imported and used in python
    config.add_extension(
        'PfyMU/read/_extensions/cwa_convert',  # note the path WITHOUT src/
        sources='src/PfyMU/read/_extensions/cwa_convert.c',  # note the path WITH src/
        libraries=['fcwa_convert']  # link the previously built fortran library
    )
    # standard C code extension that does not use a fortran library. 
    # Adding a Fortran extension follows the same syntax (numpy will do the heavy lifting for whatever compilation is required)
    config.add_extension(
        'PfyMU/read/_extensions/bin_convert',
        sources='src/PfyMU/read/_extensions/bin_convert.c'
    )

    # dealing with Cython extensions. 
    if os.environ.get('CYTHONIZE', 'False') == 'True':
        # if the environment variable was set, generate .c files from cython .pyx files. 
        # this is not necessary as the .c files are distributed with the code, but is available as an option in the off chance
        # that the .c files are not up to date
        from Cython.Build import cythonize  # only import if we need, as otherwise CYTHON isn't required as a requirement

        for pyxf in list(Path('.').rglob('*/features/lib/_cython/*.pyx')):
            cythonize(str(pyxf), compiler_directives={'language_level': 3})  # create a c file from the cython file

    # Either way, get a list of the cython .c files and add each as an extension to be compiled
    for cf in list(Path('.').rglob('*/features/lib/_cython/*.c')):
        config.add_extension(
            str(Path(*cf.parts[1:]).with_suffix('')),
            sources=[str(cf)]
        )

    # ========================
    ...

    return config
```

## Adding Tests for a new module
In order to make sure that any tests are run on installed versions of `PfyMU`, the test directory is outside the `src` directory. Again, convenience base process testing classes are available to make setting up testing easy and quick.  All testing is done using Pytest

The first step is to create a new directory for your new module under the `test` directory, and add an `__init__.py` file to allow for relative importing, the test file, and a `conftest.py` file if desired:

```
PfyMU
├── src
├── test
│   └──preprocessing
│       ├──  __init__.py
│       ├──  conftest.py
│       └── test_preprocessing.py
```

Inside `test_preprocessing.py`, import the base process testing class, set a few options, and the basic tests will be completed (when provided sample and truth data!):

```python
# test/preprocessing/test_preprocessing.py
import pytest

from ..base_conftest import *  # import BaseProcessTester, and resolve_data_path - a useful utility for making sure tests run in all 3 possible locations

from PfyMU.preprocessing import PreProcess


class TestPreProcess(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()  # make sure to call the super method

        # override specific necessary attributes
        """ resolve_data_path() takes 2 arguments
        1. the file name
        2. the module name (ie the folder name this script is in)
        """
        cls.sample_data_file = resolve_data_path('test_data.h5', 'preprocess')  
        cls.truth_data_file = resolve_data_path('test_data.h5', 'preprocess')  # can be the same file as sample data
        cls.truth_suffix = None  # if not none, means that the truth data is in the path "Truth/{truth_suffix}" in the truth h5 file
        cls.truth_data_keys = [  # list of keys to check against 
            'accel'
        ]
        cls.test_results = False  # we want to test the first return argument, not the "results"

        cls.process = PreProcess(attr1=5, attr2=10)
    
    """
    Adding additional tests, for errors, edge cases, anything that can't be accomplished with the 
    provided default structure can easily be accomplished by simply defining new functions under this class.
    Note that if the default .test method is not working for your use case, just overwrite it.
    """
    def test_error_missing_necessary_item(self):
        with pytest.raises(KeyError):
            PreProcess().predict({'not_necessary_item': 5})
```

This file would be all that is needed to test that the output `accel` values match those contained in the truth file.

### Sample/Truth data file format
The sample and truth data files are h5 files, with the below formats/keys:

```
sample.h5  # can have one or more of any of the below keys
├── time
├── accel
├── gyro
└──temperature
```

The `BaseProcessTester` class will automatically look for these keys. If you need to specify more/other keys, add the below line:
```python
@clsmethod
def setup_class(cls):
    ...
    cls.sample_data_keys.extend([
        'extra_key1',
        'extra_key2
    ])
    ...
```

The truth data is very similar. For our example (testing `accel` against its truth value):

```
truth.h5
├── Truth
│   └──accel
```

Alternatively, if `cls.truth_suffix` had been set to something else, ie `preproc`, then the structure would be as follows:

```
truth.h5
├── Truth
│   └──preproc
│       └──accel
```

the sample and truth files can be combined into 1 file:

```
sample_truth.h5
├── Truth
│   └──accel
├── time
├── accel
├── gyro
└──temperature
```

