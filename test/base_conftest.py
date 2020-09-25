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

        inp, res = self.process._predict(**data)

        if self.test_results:
            self.dict_allclose(res, truth_data, self.truth_data_keys)
        else:
            self.dict_allclose(inp, truth_data, self.truth_data_keys)

    def dict_allclose(self, pred, truth, keys):
        for key in keys:
            assert allclose(pred[key], truth[key]), f"{self.process.name} test for value ({key}) not close to truth"


@fixture(scope='module')
def get_sample_data():
    def sample_data(file, data_names):
        res = {}
        with h5py.File(file, 'r') as h5:
            for key in data_names:
                if key in h5:
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
                    truth[key] = h5[truth_key][key][()]

        return truth
    return truth_data


# MISC other utility functions
class TestRunLocationError(Exception):
    pass


def resolve_data_path(file, module=None):
    if Path.cwd().name == 'PfyMU':
        path = Path(f'test/data/{file}')
    elif Path.cwd().name == 'test':
        path = Path(f'data/{file}')
    elif module is not None:
        if Path.cwd().name == module:
            path = Path(f'../data/{file}')
    else:
        raise TestRunLocationError('tests cannot be run from this directory')

    return path
