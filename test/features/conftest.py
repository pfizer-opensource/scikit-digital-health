from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture(scope="class")
def temp_bank_file():
    file = NamedTemporaryFile(mode="w")

    yield file.name

    file.close()
