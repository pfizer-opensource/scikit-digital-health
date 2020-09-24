def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from pathlib import Path
    import os

    config = Configuration('_cython', parent_package, top_path)

    if os.getenv('CYTHONIZE', 'False') == 'True':
        from Cython.Build import cythonize

        for pxdf in list(Path('.').rglob('*/features/lib/_cython/*.pxd')):
            cythonize(str(pxdf), compiler_directives={'language_level': 3})  # create a c file from the cython file
        for pyxf in list(Path('.').rglob('*/features/lib/_cython/*.pyx')):
            if pyxf.stem == 'common':  # skip the common pyx file
                continue
            cythonize(str(pyxf), compiler_directives={'language_level': 3})  # create a c file from the cython file

    # get a list of the c files to compile
    for cf in list(Path('.').rglob('*/features/lib/_cython/*.c')):
        config.add_extension(cf.stem, sources=[str(cf)])  # Path().stem is the file name without extension

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
