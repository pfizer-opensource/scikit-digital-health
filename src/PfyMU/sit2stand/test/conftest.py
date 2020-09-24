from pytest import fixture
import h5py
from importlib import resources


# TRUTH DATA
@fixture(scope='module')
def truth_path():
    with resources.path('PfyMU.sit2stand.tests.data', 'sample.h5') as p:
        path = p
    return p


# RAW DATA
@fixture(scope='module')
def sample_data():
    with resources.path('PfyMU.sit2stand.tests.data', 'sample.h5') as p:
        with h5py.File(p, 'r') as f:
            accel = f['Sensors']['Lumbar']['Accelerometer'][()] / 9.81
            time = f['Sensors']['Lumbar']['Unix Time'][()]
    return {'time': time, 'accel': accel}
