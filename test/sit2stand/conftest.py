from pytest import fixture
import h5py
from pathlib import Path


# TRUTH DATA
@fixture(scope='module')
def truth_path():
    path = Path('../data/sample.h5')
    return path


# RAW DATA
@fixture(scope='module')
def sample_data():
    path = Path('../data/sample.h5')
    with h5py.File(path, 'r') as f:
        accel = f['Sensors']['Lumbar']['Accelerometer'][()] / 9.81
        time = f['Sensors']['Lumbar']['Unix Time'][()]

    return {'time': time, 'accel': accel}
