from pytest import fixture
import h5py
from numpy import allclose
from pathlib import Path


__all__ = ['BaseProcessTester', 'get_sample_data', 'get_truth_data', 'resolve_data_path']


# BASE TESTING CLASS
class BaseProcessTester:
    # list of names of possible sample data
    sample_data_keys = [
        'file',
        'time',
        'accel',
        'gyro',
        'temperature'
    ]

    @classmethod
    def setup_class(cls):
        cls.process = None  # the process to be tested
        cls.sample_data_file = None  # the sample data file path
        cls.truth_data_file = None  # the truth data file path
        cls.truth_data_keys = []  # the keys in the truth data to test against
        cls.truth_suffix = None   # additional path to the keys in the truth data
        cls.test_results = True   # test the results, or the data passed back out

        # tolerances for testing
        cls.atol = 1e-8
        cls.atol_time = 1e-8

    def test(self, get_sample_data, get_truth_data):
        data = get_sample_data(
            self.sample_data_file,
            self.sample_data_keys
        )

        truth_data = get_truth_data(
            self.truth_data_file,
            self.truth_data_keys,
            self.truth_suffix
        )

        res = self.process.predict(**data)

        self.dict_allclose(res, truth_data, self.truth_data_keys)

    def dict_allclose(self, pred, truth, keys):
        for key in keys:
            if key == 'time':
                ptime = pred[key] - truth[key][0]
                ttime = truth[key] - truth[key][0]
                assert allclose(ptime, ttime, atol=self.atol_time), \
                    f"{self.process._name} test for value ({key}) not close to truth"
            else:
                if isinstance(truth[key], dict):
                    for k2 in truth[key]:
                        assert allclose(
                            pred[key][k2], truth[key][k2], atol=self.atol, equal_nan=True), \
                            f"{self.process._name} test for value ({key}/{k2}) not close to truth"
                else:
                    assert allclose(pred[key], truth[key], atol=self.atol, equal_nan=True), \
                        f"{self.process._name} test for value ({key}) not close to truth"


@fixture(scope='module')
def get_sample_data():
    def sample_data(file, data_names):
        res = {}
        with h5py.File(file, 'r') as h5:
            for key in data_names:
                if key in h5:
                    if key == 'file':
                        path = resolve_data_path(h5[key][0], 'read')
                        res[key] = path
                    else:
                        res[key] = h5[key][()]
        return res
    return sample_data


@fixture(scope='module')
def get_truth_data():
    def truth_data(file, data_names, path_suffix=None):
        truth = {}
        truth_key = 'Truth' if path_suffix is None else f'Truth/{path_suffix}'

        with h5py.File(file, 'r') as h5:
            for key in data_names:
                if key in h5[truth_key]:
                    if isinstance(h5[truth_key][key], h5py.Group):
                        truth[key] = {}
                        for key2 in h5[truth_key][key]:
                            truth[key][key2] = h5[truth_key][key][key2][()]
                    else:
                        truth[key] = h5[truth_key][key][()]

        return truth
    return truth_data


# MISC other utility functions
class TestRunLocationError(Exception):
    pass


def resolve_data_path(file, module=None):

    if isinstance(file, bytes):
        file = file.decode('utf-8')

    if Path.cwd().name == 'scikit-imu':
        path = Path(f'test/data/{file}')
    elif Path.cwd().name == 'test':
        path = Path(f'data/{file}')
    elif module is not None:
        if Path.cwd().name == module:
            path = Path(f'../data/{file}')
    else:
        raise TestRunLocationError('tests cannot be run from this directory')

    return path
