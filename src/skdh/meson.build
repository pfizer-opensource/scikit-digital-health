# get the c-compiler
cc = meson.get_compiler('c')

# NumPy include directory - needed in all submodules
incdir_numpy = run_command(py3,
  [
    '-c',
    'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

# Library directory
lib_dir = '@0@/lib'.format(py3.get_path('data'))

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
#   py3.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# Add -lm for C code. For Fortran code, Meson already adds `-lm`.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

add_project_link_arguments('-Wl,-flat_namespace,-undefined,dynamic_lookup', language : 'c')


py_sources = [
    '__init__.py',
    'base.py',
    'pipeline.py',
]

py3.install_sources(
    py_sources,
    pure: false,
    subdir: 'skdh',
)

# order the subdirs so that the compilations are first for better parallelization
subdir('features')
subdir('utility')
subdir('io')
subdir('activity')
subdir('gait_old')
subdir('gait')
subdir('preprocessing')
subdir('sit2stand')
subdir('sleep')
subdir('context')
