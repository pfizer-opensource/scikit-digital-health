"""
Lukas Adamowicz
Pfizer DMTI 2020
"""

from numpy.f2py import capi_maps

# MODIFY the f2c maps for types
capi_maps.f2cmap_all["real"].update(real32="float", real64="double")
capi_maps.f2cmap_all["integer"].update(
    int8="signed_char", int16="int", int32="long", int64="long_long"
)


# package setup


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("skimu", parent_package, top_path)

    # sub packages (ADD NEW PACKAGES HERE)
    # ==============================
    config.add_subpackage("features")
    config.add_subpackage("gait")
    config.add_subpackage("read")
    config.add_subpackage("sit2stand")
    # ==============================

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
