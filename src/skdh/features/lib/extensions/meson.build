py3.install_sources(
    ['__init__.py'],
    pure: false,
    subdir: 'skdh/features/lib/extensions',
)

fort_features_sources = [
    'ffeatures.f95',
    'real_fft.f95',
    'f_rfft.f95',
    'sort.f95',
    'utility.f95',
]

fort_features_lib = static_library(
    'fort_features',
    fort_features_sources,
    c_args: numpy_nodepr_api,
    include_directories: [inc_np],
)

features_sources = [
    'entropy',
    'frequency',
    'misc_features',
    'smoothness',
    'statistics',
    '_utility',
]

foreach feat_source: features_sources
    py3.extension_module(
        feat_source,
        '@0@.c'.format(feat_source),
        include_directories: [inc_np],
        link_with: [
            fort_features_lib,
        ],
        link_language: 'fortran',
        c_args: numpy_nodepr_api,
        install: true,
        subdir: 'skdh/features/lib/extensions',
    )
endforeach

