py_sources = [
    '__init__.py',
    'entropy.py',
    'frequency.py',
    'misc.py',
    'moments.py',
    'multi_axis.py',
    'smoothness.py',
    'statistics.py',
    'wavelet.py',
]

py3.install_sources(
    py_sources,
    pure: false,
    subdir: 'skdh/features/lib',
)

if is_not_rtd_build
    subdir('extensions')
endif
