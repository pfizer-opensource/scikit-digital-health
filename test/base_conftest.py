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
        cls.process = None
        cls.sample_data_file = None
        cls.truth_data_file = None
        cls.truth_data_keys = []
        cls.truth_suffix = None

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
