py3.install_sources(
    [
        '__init__.py',
    ],
    pure: false,
    subdir: 'skdh/io/_extensions',
)

read_lib = static_library(
    'read',
    [
        'read_axivity.f95',
        'read_geneactiv.c',
    ],
    c_args: numpy_nodepr_api,
    include_directories: [inc_np],
#    dependencies: py3_dep,
)

py3.extension_module(
    'read',
    sources: ['pyread.c'],
    include_directories: [inc_np],
    c_args: numpy_nodepr_api,
    link_with: [read_lib],
    link_language: 'fortran',
    install: true,
    subdir: 'skdh/io/_extensions',
)
