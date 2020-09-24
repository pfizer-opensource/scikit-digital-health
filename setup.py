import os
import sys
import textwrap
import warnings


PACKAGE_NAME = 'PfyMU'

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU GPL v3
Programming Language :: Fortran 95
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: MacOS
"""


REQUIREMENTS = [
    'cython>=0.29.14',
    'numpy>=1.17.2',
    'scipy>=1.3.1',
    'pandas>=0.23.4',
    'lightgbm>=2.3.0'
]


def parse_setuppy_commands():
    """
    Check the commands and respond appropriately.  Disable broken commands.

    Return a boolean value for whether or not to run the build or not.
    """
    args = sys.argv[1:]

    if not args:
        # user forgot to give an argument. Let setuptools handle that
        return True

    # setup for the cythonize command
    if 'cythonize' in args:
        os.environ['CYTHONIZE'] = 'True'

    info_commands = ['--help-commands', '--name', '--version', '-V', '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact', '--contact-email', '--url', '--license',
                     '--description', '--long-description', '--platforms', '--classifiers', '--keywords', '--provides',
                     '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py', 'build_clib', 'build_scripts', 'bdist_wheel',
                     'bdist_rpm', 'bdist_wininst', 'bdist_msi', 'bdist_mpkg')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent(f"""
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup.py install`:
              - `pip install .`       (from a git repo or downloaded source
                                       release)
            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent(f"""
            Help
            -------------------
            To install {PACKAGE_NAME} from here with reliable uninstall, we recommend
            that you use `pip install .`.
            For help with build/installation issues, please ask on the
            github repository.

            Setuptools commands help
            ------------------------
            """))
        return False

    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test=f"""
            `setup.py test` is not supported.
            """,
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        upload_docs="`setup.py upload_docs` is not supported",
        easy_install="`setup.py easy_install` is not supported",
        clean="""
            `setup.py clean` is not supported, use one of the following instead:
              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        check="`setup.py check` is not supported",
        register="`setup.py register` is not supported",
        bdist_dumb="`setup.py bdist_dumb` is not supported",
        bdist="`setup.py bdist` is not supported",
    )

    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                    'register', 'check', 'install_data', 'install_headers',
                    'install_lib', 'install_scripts',):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    warnings.warn("Unrecognized setuptools command ('{}'), proceeding with "
                  "generating sources and expanding templates".format(' '.join(sys.argv[1:])))
    return True


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    # add the PfyMU subpackage, which will list all the additional subpackages
    config.add_subpackage(PACKAGE_NAME, subpackage_path='src')

    config.get_version(f'src/{PACKAGE_NAME}/version.py')

    return config


def setup_package():
    if sys.version_info < (3, 7):  # check if using version 3.6
        REQUIREMENTS.append('importlib_resources')

    with open("README.md", "r") as fh:
        long_description = fh.read()

    with open(f'src/{PACKAGE_NAME}/version.py') as fid:
        vers = fid.readlines()[-1].split()[-1].strip("\"'")

    metadata = dict(
        name=PACKAGE_NAME,
        maintainer='Lukas Adamowicz',
        maintainer_email='lukas.adamowicz@pfizer.com',
        description='Python general purpose IMU analysis and processing package.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/PfizerRD/PfyMU',
        license='GNU GPL v3',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=['Mac OS-X'],
        setup_requires=REQUIREMENTS,
        install_requires=REQUIREMENTS,
        python_requires='>=3.6',
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # raise errors for unsupported commands, improve help output, etc
        run_build = parse_setuppy_commands()

    from setuptools import setup
    if run_build:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration
    else:
        metadata['version'] = vers

    setup(**metadata)


if __name__ == '__main__':
    setup_package()




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

extensions = []
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
"""
