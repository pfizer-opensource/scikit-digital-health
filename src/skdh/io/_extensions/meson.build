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
        'utility.f95',
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

# try to find the zip library. If its not found we won't generate the extension
zip_dep = cc.find_library('zip', required: false, dirs: lib_dir)

if zip_dep.found()
    # get the include directory for conda
    conda_incl = include_directories(
        '@0@/include'.format(
            py3.get_path('data')
        )
    )

    # create the reader library
    gt3x_lib = static_library(
        'gt3x',
        'gt3x.c',
        c_args: numpy_nodepr_api,
        include_directories: [inc_np, conda_incl],
    )
    # create the extension
    py3.extension_module(
        'gt3x_convert',
        'pygt3x_convert.c',
        include_directories: [inc_np, conda_incl],
        dependencies: zip_dep,
        link_with: [gt3x_lib],
        c_args: numpy_nodepr_api,
        install: true,
        subdir: 'skdh/io/_extensions',
    )
else
    warning('Library `zip` not found, not building the GT3x reader extension.')
endif
