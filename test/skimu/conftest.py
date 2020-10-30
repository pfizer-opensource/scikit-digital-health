from tempfile import NamedTemporaryFile

from pytest import fixture


@fixture(scope='module')
def pipeline_file():
    ntf = NamedTemporaryFile(mode='a')

    yield ntf.name

    ntf.close()


@fixture()
def pipe_file2():
    ntf = NamedTemporaryFile(mode='a')

    yield ntf.name

    ntf.close()


@fixture(scope='module')
def gait_res_file():
    ntf = NamedTemporaryFile(mode='a')

    yield ntf.name

    ntf.close()
