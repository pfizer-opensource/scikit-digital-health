py3.install_sources(
    [
        '__init__.py',
    ],
    pure: false,
    subdir: 'skdh/utility/_extensions',
)

movstat_lib = static_library(
    'fmoving_statistics',
    sources: [
        'sort.f95',
        'moving_moments.f95',
        'median_heap.f95',
    ],
    c_args: numpy_nodepr_api,
    include_directories: [inc_np],
    dependencies: py3_dep,
)

py3.extension_module(
    'moving_statistics',
    sources: [
        'moving_statistics.c',
    ],
    include_directories: [inc_np],
    dependencies: py3_dep,
    link_with: [movstat_lib],
    link_language: 'fortran',
    c_args: numpy_nodepr_api,
    install: true,
    subdir: 'skdh/utility/_extensions',
)