project(
    'scikit-digital-health',
    'c',
    version: '0.16.12',
    license: 'MIT',
    meson_version: '>=1.1',
)

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation()
py3_dep = py3.dependency()

is_not_rtd_build = get_option('is_not_rtd_build')
message('Is not RTD build:', is_not_rtd_build)

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false, required: is_not_rtd_build)

# debugging printing
# add_project_arguments('-DDEBUG', language: 'c')

subdir('src/skdh')
