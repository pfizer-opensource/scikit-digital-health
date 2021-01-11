from pytest import fixture
import numpy as np
from pandas import DataFrame
import h5py

from ..base_conftest import resolve_data_path


SAMPLE_DATA_PATH = resolve_data_path('sample_accelerometer.h5', 'features')
FEATURES_TRUTH_PATH = resolve_data_path('features_truth.h5', 'features')


@fixture(scope="module")
def fs():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, "r") as f:
        ret = f.attrs.get("Sampling rate")

    return ret


@fixture(scope='package')
def x():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        ret = f['Accelerometer'][:, 0]

    return ret


@fixture(scope='package')
def y():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        ret = f['Accelerometer'][:, 1]

    return ret


@fixture(scope='package')
def z():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        ret = f['Accelerometer'][:, 2]

    return ret


@fixture(scope='package')
def acc():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        ret = f['Accelerometer'][()]

    return ret


@fixture(scope='package')
def win_acc():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        acc = f['Accelerometer'][()]

    # broadcast the acceleration into windows (with the same samples)
    ret = np.broadcast_to(acc, (4, ) + acc.shape)
    return ret


@fixture(scope='package')
def df_acc():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        acc = f['Accelerometer'][()]
    ret = DataFrame(data=acc, columns=['x', 'y', 'z'])

    return ret


@fixture(scope='package')
def get_1d_truth():
    def get_1d(name):
        path = FEATURES_TRUTH_PATH
        with h5py.File(path, 'r') as f:
            x_, y_, z_ = f[name][()].flatten()

        xtr = x_.reshape((1, 1, 1))
        ytr = y_.reshape((1, 1, 1))
        ztr = z_.reshape((1, 1, 1))

        return xtr, ytr, ztr
    return get_1d


@fixture(scope='package')
def get_2d_truth():
    def get_2d(name):
        path = FEATURES_TRUTH_PATH
        with h5py.File(path, 'r') as f:
            truth = f[name][()].reshape((1, 3))
        return truth

    return get_2d


@fixture(scope='package')
def get_3d_truth():
    def get_3d(name):
        path = FEATURES_TRUTH_PATH
        with h5py.File(path, 'r') as f:
            truth = f[name][()].reshape((1, 3))
        return np.broadcast_to(truth, (4, 3))

    return get_3d


@fixture(scope='package')
def get_dataframe_truth():
    def get_df(name):
        path = FEATURES_TRUTH_PATH
        with h5py.File(path, 'r') as f:
            truth = f[name][()]

        return truth, [f'{name}_x', f'{name}_y', f'{name}_z']
    return get_df
