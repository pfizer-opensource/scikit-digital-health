from pytest import fixture
import h5py
from sys import version_info

if version_info >= (3, 7):
    from importlib import resources
else:
    import importlib_resources


# TRUTH DATA
@fixture(scope='module')
def truth_path():
    if version_info >= (3, 7):
        with resources.path('PfyMU.sit2stand.tests.data', 'sample.h5') as p:
            path = p
    else:
        with importlib_resources.path('PfyMU.sit2stand.tests.data', 'sample.h5') as p:
            path = p
    return path


# RAW DATA
@fixture(scope='module')
def sample_data():
    if version_info >= (3, 7):
        with resources.path('PfyMU.sit2stand.tests.data', 'sample.h5') as p:
            with h5py.File(p, 'r') as f:
                accel = f['Sensors']['Lumbar']['Accelerometer'][()] / 9.81
                time = f['Sensors']['Lumbar']['Unix Time'][()]
    else:
        with importlib_resources.path('PfyMU.sit2stand.tests.data', 'sample.h5') as p:
            with h5py.File(p, 'r') as f:
                accel = f['Sensors']['Lumbar']['Accelerometer'][()] / 9.81
                time = f['Sensors']['Lumbar']['Unix Time'][()]
    return {'time': time, 'accel': accel}
