"""
Lukas Adamowicz
Pfizer DMTI 2020
"""


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('_extensions', parent_package, top_path)

    # CWA converter
    config.add_library('fcwa_convert', sources=['cwa_convert.f95'])
    config.add_extension('cwa_convert', sources=['cwa_convert.c'], libraries=['fcwa_convert'])

    # BIN (GeneActiv) converter
    config.add_extension('bin_convert', sources='bin_convert.c')

    # GT3X (Actigraph) converter
    config.add_library('gt3x', sources=['gt3x.c'])
    config.add_extension(
        'read_gt3x',
        sources=['read_gt3x.c'],
        libraries=['gt3x', 'zip']
    )

    return config


if __name__ == '_main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
