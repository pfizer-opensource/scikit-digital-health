import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('jerk', parent_package, top_path)

    # Configure pocketfft_internal
    config.add_extension('_jerkmetric',
                         sources=['jerk.c']
                         )

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)