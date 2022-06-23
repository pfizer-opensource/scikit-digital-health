import sysconfig
import os
import setuptools


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    # config.set_options(
    #     ignore_setup_xxx_py=True,
    #     assume_default_configuration=True,
    #     delegate_options_to_subpackages=True,
    #     quiet=False,
    # )

    # EXTENSIONS
    # ========================
    # if the environment variable is NOT present, build the extensions
    build_extensions = "BUILD_SKDH_EXTENSIONS" not in os.environ
    if build_extensions:
        # check if on windows and have a conda prefix path
        if (os.name == 'nt') and ('CONDA_PREFIX' in os.environ):
            # add an extra include path for where conda puts libraries in windows
            config.add_include_dirs([f"{os.environ['CONDA_PREFIX']}\\Library\\Include"])

        # add extension paths all at once
        config.add_include_dirs(sysconfig.get_path("data") + os.sep + "include")
        # UTILITY
        config.add_library(
            "fmoving_statistics",
            sources=[
                "src/skdh/utility/_extensions/sort.f95",
                "src/skdh/utility/_extensions/moving_moments.f95",
                "src/skdh/utility/_extensions/median_heap.f95"
            ]
        )
        config.add_extension(
            "skdh/utility/_extensions/moving_statistics",
            sources="src/skdh/utility/_extensions/moving_statistics.c",
            libraries=["fmoving_statistics"],
        )

        # Read library
        config.add_library(
            "read",
            sources=[
                # utility HAS to be first so that read_axivity can use the module it compiles
                "src/skdh/io/_extensions/utility.f95",
                "src/skdh/io/_extensions/read_axivity.f95",
                "src/skdh/io/_extensions/read_geneactiv.c",
            ],
        )
        config.add_extension(
            "skdh/io/_extensions/read",
            sources="src/skdh/io/_extensions/pyread.c",
            libraries=["read"],
        )

        # gt3x (actigraph)  needs its own library for some reason
        config.add_library("gt3x", sources="src/skdh/io/_extensions/gt3x.c")
        config.add_extension(
            "skdh/io/_extensions/gt3x_convert",
            sources=["src/skdh/io/_extensions/pygt3x_convert.c"],
            libraries=["gt3x", "zip"],
        )

        # Fortran/C feature extensions
        config.add_library(
            "ffeatures",
            sources=[
                "src/skdh/features/lib/extensions/ffeatures.f95",
                "src/skdh/features/lib/extensions/real_fft.f95",
                "src/skdh/features/lib/extensions/f_rfft.f95",
                "src/skdh/features/lib/extensions/sort.f95",
                "src/skdh/features/lib/extensions/utility.f95",
            ],
        )
        for ext in [
            "entropy",
            "frequency",
            "misc_features",
            "smoothness",
            "statistics",
            "_utility",
        ]:
            config.add_extension(
                f"skdh/features/lib/extensions/{ext}",
                sources=[f"src/skdh/features/lib/extensions/{ext}.c"],
                libraries=["ffeatures"],
            )

    # ========================
    # DATA FILES
    # ========================
    config.add_data_files(
        ("skdh", "src/skdh/VERSION"),
        ("skdh/io/_extensions", "src/skdh/io/_extensions/read_binary_imu.h"),
        ("skdh/io/_extensions", "src/skdh/io/_extensions/gt3x.h"),
        ("skdh/gait/model", "src/skdh/gait/model/final_features.json"),
        (
            "skdh/gait/model",
            "src/skdh/gait/model/lgbm_gait_classifier_no-stairs_50hz.lgbm",
        ),
        (
            "skdh/gait/model",
            "src/skdh/gait/model/lgbm_gait_classifier_no-stairs_20hz.lgbm",
        ),
    )
    # ========================

    return config


def setup_package():
    from numpy.distutils.core import setup

    setup_kwargs = {"configuration": configuration}

    setup(**setup_kwargs)


if __name__ == "__main__":
    setup_package()
