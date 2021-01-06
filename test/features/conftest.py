from pytest import fixture
import numpy as np
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

        assert np.allclose(x_pred, x_truth)
        assert np.allclose(y_pred, y_truth)
        assert np.allclose(z_pred, z_truth)

    def test_2d_ndarray(self, fs, acc, get_2d_truth):
        truth = get_2d_truth(self.feature.__class__.__name__)

        try:
            pred = self.feature.compute(acc, fs, axis=0)
        except TypeError:
            pred = self.feature.compute(acc, axis=0)

        assert np.allclose(pred, truth)

    def test_3d_ndarray(self, fs, win_acc, get_3d_truth):
        truth = get_3d_truth(self.feature.__class__.__name__)

        try:
            pred = self.feature.compute(win_acc, fs, axis=1)
        except TypeError:
            pred = self.feature.compute(win_acc, axis=1)

        assert np.allclose(pred, truth)

    def test_dataframe(self, fs, df_acc, get_dataframe_truth):
        df_truth, cols = get_dataframe_truth(self.feature.__class__.__name__)

        try:
            df_pred = self.feature.compute(df_acc, fs)
        except TypeError:
            df_pred = self.feature.compute(df_acc)

        assert np.allclose(df_pred, df_truth)

    def test_dataframe_with_cols(self, fs, df_acc, get_dataframe_truth):
        df_truth, cols = get_dataframe_truth(self.feature.__class__.__name__)

        try:
            df_pred = self.feature[[0, 2]].compute(df_acc, fs)
        except TypeError:
            df_pred = self.feature[[0, 2]].compute(df_acc)

        assert np.allclose(df_pred, df_truth[0, [0, 2]])


@fixture(scope='package')
def fs():
    path = SAMPLE_DATA_PATH
    with h5py.File(path, 'r') as f:
        ret = f.attrs.get('Sampling rate')

    return ret


@fixture(scope="package")
def random_wave():
    class RandomWave:
        __slots__ = ("x", "y", "fs", "freq1", "freq2", "amp1", "amp2", "slope", "itcpt",
                     "noise_amp", "axis")

        def __init__(self, fs, ndim):
            shape = (ndim,) * (ndim - 1)

            self.fs = fs

            self.freq1 = np.around(np.random.rand(*shape) * 5, 2)
            self.freq2 = 5 + np.around(np.random.rand(*shape) * 5, 2)
            self.amp1 = np.around(np.random.rand(*shape), 2)
            self.amp2 = np.around(np.random.rand(*shape) * 0.5, 2)
            self.slope = np.around(np.random.rand(*shape) * 0.15, 3)
            self.itcpt = np.sign(np.random.rand(*shape) - 0.5) * np.around(np.random.rand(*shape), 3)

            self.x = np.arange(0, 10 + 0.5 / fs, 1 / fs)[np.newaxis]

            self.y = None
            self.get_y()

            self.axis = -1

        def get_y(self):
            nax = np.newaxis
            p2 = 2 * np.pi

            self.y = (self.amp1[..., nax] * np.sin(p2 * self.freq1[..., nax] * self.x)
                      + self.amp2[..., nax] * np.sin(p2 * self.freq2[..., nax] * self.x)
                      + self.slope[..., nax] * self.x + self.itcpt[..., nax])
    return RandomWave


@fixture
def random_linear():
    class Linear:
        __slots__ = ("x", "y", "fs", "slope", "itcpt", "axis")

        def __init__(self, fs, ndim):
            shape = (ndim,) * (ndim - 1)

            self.fs = fs
            self.slope = np.around(np.random.rand(*shape) * 0.15, 3)
            self.itcpt = np.sign(np.random.rand(*shape) - 0.5) * np.around(np.random.rand(*shape), 3)

            self.x = np.arange(0, 10, 1 / fs)

            self.y = self.slope[..., np.newaxis] * self.x + self.itcpt[..., np.newaxis]

            self.axis = -1

    return Linear

    
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


@fixture(scope='package')
def bank_2d_truth():
    """
    bank + Mean()
    bank + Range()[['x', 'z']]
    bank + JerkMetric(normalize=True)
    bank + Range()['y']
    """
    truth = np.zeros((1, 9))

    path = FEATURES_TRUTH_PATH
    with h5py.File(path, 'r') as f:
        truth[0, :3] = f['Mean']
        truth[0, [3, 8, 4]] = f['Range']
        truth[0, 5:8] = f['JerkMetric']

    return truth
