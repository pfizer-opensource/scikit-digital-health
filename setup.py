import sys
import sysconfig
import os
import textwrap
import warnings


def parse_setuppy_commands():
    """
    Check the commands and respond appropriately.  Disable broken commands.

    Return a boolean value for whether or not to run the build or not.
    """
    args = sys.argv[1:]

    if not args:
        # user forgot to give an argument. Let setuptools handle that
        return True

    info_commands = [
        "--help-commands",
        "--name",
        "--version",
        "-V",
        "--fullname",
        "--author",
        "--author-email",
        "--maintainer",
        "--maintainer-email",
        "--contact",
        "--contact-email",
        "--url",
        "--license",
        "--description",
        "--long-description",
        "--platforms",
        "--classifiers",
        "--keywords",
        "--provides",
        "--requires",
        "--obsoletes",
    ]

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = (
        "develop",
        "sdist",
        "build",
        "build_ext",
        "build_py",
        "build_clib",
        "build_scripts",
        "bdist_wheel",
        "bdist_rpm",
        "bdist_wininst",
        "bdist_msi",
        "bdist_mpkg",
    )

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if "install" in args:
        print(
            textwrap.dedent(
                """
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup.py install`:
              - `pip install .`       (from a git repo or downloaded source
                                       release)
            """
            )
        )
        return True

    if "--help" in args or "-h" in sys.argv[1]:
        print(
            textwrap.dedent(
                """
            Help
            -------------------
            To install inertial-sensor-routines from here with reliable 
            uninstall, we recommend that you use `pip install .`.
            For help with build/installation issues, please ask on the
            github repository.

            Setuptools commands help
            ------------------------
            """
            )
        )
        return False

    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test="""
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

    bad_commands["nosetests"] = bad_commands["test"]
    for command in (
        "upload_docs",
        "easy_install",
        "bdist",
        "bdist_dumb",
        "register",
        "check",
        "install_data",
        "install_headers",
        "install_lib",
        "install_scripts",
    ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(
                textwrap.dedent(bad_commands[command])
                + "\nAdd `--force` to your command to use it anyway if you "
                "must (unsupported).\n"
            )
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ["egg_info", "install_egg_info", "rotate"]
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    warnings.warn(
        "Unrecognized setuptools command ('{}'), proceeding with "
        "generating sources and expanding "
        "templates".format(" ".join(sys.argv[1:]))
    )
    return True


MAINTAINERS = [
    "Pfizer DMTI Analytics",
    "Lukas Adamowicz",
    "Yiorgos Christakis",
]

MAINTAINER_EMAILS = [
    "lukas.adamowicz@pfizer.com",
    "Yiorgos.Christakis@pfizer.com",
]


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


with open("requirements.txt", "r") as f:
    _req = f.readlines()
REQUIREMENTS = [i.strip() for i in _req]

if sys.version_info < (3, 7):
    REQUIREMENTS.append("importlib_resources")


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=False
    )

    # EXTENSIONS
    # ========================
    # add extension paths all at once
    config.add_include_dirs(sysconfig.get_path("data") + os.sep + "include")
    # UTILITY
    config.add_library(
        "fmoving_moments", sources="src/skimu/utility/_extensions/moving_moments.f95"
    )
    config.add_extension(
        "skimu/utility/_extensions/moving_moments",
        sources="src/skimu/utility/_extensions/moving_moments.c",
        libraries=["fmoving_moments"]
    )
    config.add_extension(
        "skimu/utility/_extensions/moving_median",
        sources="src/skimu/utility/_extensions/moving_median.c",
        libraries=["gsl"]
    )
    # Read library
    config.add_library(
        "read",
        sources=[
            # utility HAS to be first so that read_axivity can use the module it compiles
            "src/skimu/read/_extensions/utility.f95",
            "src/skimu/read/_extensions/read_axivity.f95",
            "src/skimu/read/_extensions/read_geneactiv.c",
        ]
    )
    config.add_extension(
        "skimu/read/_extensions/read",
        sources="src/skimu/read/_extensions/pyread.c",
        libraries=["read"],
    )

    # gt3x (actigraph)  needs its own library for some reason
    config.add_library("gt3x", sources="src/skimu/read/_extensions/gt3x.c")
    config.add_extension(
        "skimu/read/_extensions/gt3x_convert",
        sources=["src/skimu/read/_extensions/pygt3x_convert.c"],
        libraries=["gt3x", "zip"]
    )

    # Fortran/C feature extensions
    config.add_library(
        "ffeatures",
        sources=[
            "src/skimu/features/lib/extensions/ffeatures.f95",
            "src/skimu/features/lib/extensions/real_fft.f95",
            "src/skimu/features/lib/extensions/f_rfft.f95",
            "src/skimu/features/lib/extensions/sort.f95",
            "src/skimu/features/lib/extensions/utility.f95",
        ]
    )
    for ext in [
        "entropy",
        "frequency",
        "misc_features",
        "smoothness",
        "statistics",
        "_utility"
    ]:
        config.add_extension(
            f"skimu/features/lib/extensions/{ext}",
            sources=[f"src/skimu/features/lib/extensions/{ext}.c"],
            libraries=["ffeatures"]
        )

    # ========================
    # DATA FILES
    # ========================
    config.add_data_files(
        ("skimu/gait/model", "src/skimu/gait/model/final_features.json"),
        (
            "skimu/gait/model",
            "src/skimu/gait/model/lgbm_gait_classifier_no-stairs_50hz.lgbm",
        ),
        (
            "skimu/gait/model",
            "src/skimu/gait/model/lgbm_gait_classifier_no-stairs_20hz.lgbm",
        )
    )
    # ========================

    config.get_version("src/skimu/version.py")

    return config


def setup_package():
    from setuptools import find_packages

    with open("README.md", "r") as fh:
        long_description = fh.read()
    with open("src/skimu/version.py") as fid:
        vers = fid.readlines()[-1].split()[-1].strip("\"'")

    setup_kwargs = dict(
        name="scikit-imu",
        maintainer=MAINTAINERS,
        maintainer_email=MAINTAINER_EMAILS,
        description="Python general purpose IMU data processing package.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # download_url="https:/pypi.org/skimu",  # download link, likely PyPi
        # project_urls={
        #     "Documentation": "https://skimu.readthedocs.io./en/latest/"
        # },
        packages=find_packages("src"),
        package_dir={"": "src"},
        license="GNU GPL v3",
        python_requires=">=3.6",
        setup_requires=REQUIREMENTS,
        install_requires=REQUIREMENTS,
        classifiers=CLASSIFIERS,
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove("--force")
    else:
        # raise errors for unsupported commands, improve help output, etc
        run_build = parse_setuppy_commands()

    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup

        setup_kwargs["configuration"] = configuration
    else:
        setup_kwargs["version"] = vers

    setup(**setup_kwargs)


if __name__ == "__main__":
    setup_package()
