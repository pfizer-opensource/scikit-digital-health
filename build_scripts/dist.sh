#!/usr/bin/env python

# Builds both a wheel and a distribution
python -m build
# If building only a wheel
python -m build --wheel
# if building only a source dist
python -m build --sdist

#########
# delocate for mac wheels
# list dependencies
delocate-listdeps --depending dist/*.whl
# fix dependencies
delocate-wheel -w fixed_wheels -v dist/*.whl

##########
# Uploading files
python -m twine upload --skip-existing --repository skdh dist/*.tar.gz fixed_wheels/*