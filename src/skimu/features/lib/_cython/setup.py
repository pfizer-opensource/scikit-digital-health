def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from pathlib import Path
    import os

    config = Configuration('_cython', parent_package, top_path)

    if os.getenv('CYTHONIZE', 'False') == 'True':
        from Cython.Build import cythonize

        # create a c file from the cython file
        for pxdf in list(Path('.').rglob('*/features/lib/_cython/*.pxd')):
            cythonize(pxdf.name, compiler_directives={'language_level': 3})
        for pyxf in list(Path('.').rglob('*/features/lib/_cython/*.pyx')):
            if pyxf.stem == 'common':  # skip the common pyx file
                continue
            # create a c file from the cython file
            cythonize(pyxf.name, compiler_directives={'language_level': 3})

    # get a list of the c files to compile
    for cf in list(Path('.').rglob('*/features/lib/_cython/*.c')):
        # Path().stem is the file name without extension
        config.add_extension(cf.stem, sources=[cf.name])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
