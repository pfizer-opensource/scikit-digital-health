from pytest import fixture
from numpy import allclose, broadcast_to, zeros
from pandas.testing import assert_frame_equal
from pandas import DataFrame
import h5py

from ..base_conftest import resolve_data_path


SAMPLE_DATA_PATH = resolve_data_path('sample_accelerometer.h5', 'features')
FEATURES_TRUTH_PATH = resolve_data_path('features_truth.h5', 'features')


class BaseTestFeature:
    def test_1d_ndarray(self, fs, x, y, z, get_1d_truth):
        x_truth, y_truth, z_truth = get_1d_truth(self.feature.__class__.__name__)

        try:
            x_pred = self.feature.compute(x, fs)
            y_pred = self.feature.compute(y, fs)
            z_pred = self.feature.compute(z, fs)
        except TypeError:
            x_pred = self.feature.compute(x)
            y_pred = self.feature.compute(y)
            z_pred = self.feature.compute(z)

        assert allclose(x_pred, x_truth)
        assert allclose(y_pred, y_truth)
        assert allclose(z_pred, z_truth)

    def test_2d_ndarray(self, fs, acc, get_2d_truth):
        truth = get_2d_truth(self.feature.__class__.__name__)

        try:
            pred = self.feature.compute(acc, fs)
        except TypeError:
            pred = self.feature.compute(acc, axis=0)

        assert allclose(pred, truth)

    def test_3d_ndarray(self, fs, win_acc, get_3d_truth):
        truth = get_3d_truth(self.feature.__class__.__name__)

        pred = self.feature.compute(win_acc, fs)

        assert allclose(pred, truth)

    def test_dataframe(self, fs, df_acc, get_dataframe_truth):
        df_truth = get_dataframe_truth(self.feature.__class__.__name__)

        df_pred = self.feature.compute(df_acc, fs)

        assert_frame_equal(df_pred, df_truth)


@fixture(scope='package')
def fs():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        ret = f.attrs.get('Sampling rate')

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
    ret = broadcast_to(acc, (4, ) + acc.shape)
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
        return broadcast_to(truth, (4, 3))

    return get_3d


@fixture(scope='package')
def get_dataframe_truth():
    def get_df(name):
        path = FEATURES_TRUTH_PATH
        with h5py.File(path, 'r') as f:
            truth = f[name][()]

        return DataFrame(data=truth, columns=[f'{name}_x', f'{name}_y', f'{name}_z'])
    return get_df


@fixture(scope='package')
def bank_2d_truth():
    """
    bank + Mean()
    bank + Range()[['x', 'z']]
    bank + JerkMetric(normalize=True)
    bank + Range()['y']
    """
    truth = zeros((1, 9))

    path = FEATURES_TRUTH_PATH
    with h5py.File(path, 'r') as f:
        truth[0, :3] = f['Mean']
        truth[0, [3, 8, 4]] = f['Range']
        truth[0, 5:8] = f['JerkMetric']

    return truth
