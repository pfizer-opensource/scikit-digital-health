import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(parent_package, top_path)
    
    config.add_library('facorr', sources=['acorr.f95'])
    config.add_extension('acorr', sources=['acorr.c'], libraries=['facorr'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)