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

        """
        FINALLY: the return of the _predict function needs to be 2 items
        1. a dictionary of the input to _predict, and anything that might be needed in other stages
        2. anything else that needs to be returned, results, etc
        """
        # for this specific case, the goal is to modify the acceleration (hence preprocessing, the modified version needs to be returned in the input dictionary)
        # additionally, there are no specific results from this step, so None is returned as the second argument
        return dict(time=time, accel=accel2, **kwargs), None
```

### 3. Make sure everything is setup/imported
Make sure all imports are handled in `src/PfyMU/preprocessing/__init__.py`, as well as adding `preprocessing` imports to the `src/PfyMU/__init__.py`.

### 4. Make any additions to setup.py
If you don't have any data files (any non python files that need to be distributed with the package), or low level (c, cython, or fortran) extensions, everything should be good for the actual module, and you can skip the next steps

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

    config.add_data_dir('src/PfyMU/preprocessing/data')  # alternatively add this directory, any files/folders under this directory will be added recursively
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
        

