from pytest import fixture
from numpy import allclose, broadcast_to
from pandas.testing import assert_frame_equal
from pandas import DataFrame
from importlib import resources
import h5py


class TestFeature:
    def test_1d_ndarray(self, fs, x, y, z, get_1d_truth):
        x_truth, y_truth, z_truth = get_1d_truth(self.feature._name)

        x_pred = self.feature.compute(x, fs)
        y_pred = self.feature.compute(y, fs)
        z_pred = self.feature.compute(z, fs)

        assert allclose(x_pred, x_truth)
        assert allclose(y_pred, y_truth)
        assert allclose(z_pred, z_truth)

    def test_2d_ndarray(self, fs, acc, get_2d_truth):
        truth = get_2d_truth(self.feature._name)

        pred = self.feature.compute(acc, fs)

        assert allclose(pred, truth)

    def test_3d_ndarray(self, fs, win_acc, get_3d_truth):
        truth = get_3d_truth(self.feature._name)

        pred = self.feature.compute(win_acc, fs)

        assert allclose(pred, truth)

    def test_dataframe(self, fs, df_acc, get_dataframe_truth):
        df_truth = get_dataframe_truth

        df_pred = self.feature.compute(df_acc, fs)

        assert_frame_equal(df_pred, df_truth)


@fixture(scope='package')
def fs():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            ret = f.attrs.get('Sampling rate')
    return ret


@fixture(scope='package')
def x():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            ret = f['Accelerometer'][:, 0]
    return ret


@fixture(scope='package')
def y():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            ret = f['Accelerometer'][:, 1]
    return ret


@fixture(scope='package')
def z():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            ret = f['Accelerometer'][:, 2]
    return ret


@fixture(scope='package')
def acc():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            ret = f['Accelerometer'][()]
    return ret


@fixture(scope='package')
def win_acc():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            acc = f['Accelerometer'][()]

    # broadcast the acceleration into windows (with the same samples)
    ret = broadcast_to(acc, (4, ) + acc.shape)
    return ret


@fixture(scope='package')
def df_acc():
    with resources.path('PfyMU.tests.data', 'sample_accelerometer.h5') as path:
        with h5py.File(path, 'r') as f:
            acc = f['Accelerometer'][:, 1]
    ret = DataFrame(data=acc, columns=['x', 'y', 'z'])
    return ret

