py3.install_sources(
    [
        '__init__.py',
    ],
    pure: false,
    subdir: 'skdh/preprocessing/_extensions',
)

window_lib = static_library(
    'window_days',
    [
        'window_days.c',
    ],
    c_args: numpy_nodepr_api,
)

py3.extension_module(
    'pywindow',
    sources: ['pywindow.c'],
    include_directories: [inc_np],
    c_args: numpy_nodepr_api,
    link_with: [window_lib],
    install: true,
    subdir: 'skdh/preprocessing/_extensions',
)