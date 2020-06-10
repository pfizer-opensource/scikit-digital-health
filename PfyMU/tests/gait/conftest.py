from pytest import fixture
from tempfile import TemporaryDirectory as TempDir
from pathlib import Path
import h5py
import numpy as np


@fixture()
def sample_datasets():
    study1_td = TempDir()
    study1_path = Path(study1_td)
    study2_td = TempDir()
    study2_path = Path(study2_td)

    # study 1
    for k in range(2):
        with h5py.File(study1_path / f'subject_{k}.h5', 'w') as f:
            for j in range(3):
                ag = f.create_group(f'activity{j}')
                ag.attrs.create('Gait Label', 1 if j == 1 else 0)

                for i in range(2):
                    agt = ag.create_group(f'Trial {i}')
                    agt.attrs.create('Sampling rate', 100.0)

                    agt.create_dataset('Accelerometer', data=np.random.rand(1500, 3) + np.array([[0, 0, 1]]))

    # study 2
    for k in range(2):
        with h5py.File(study2_path / f'subject_{k}.h5', 'w') as f:
            for j in range(3):
                ag = f.create_group(f'activity{j}')
                ag.attrs.create('Gait Label', 1 if j == 1 else 0)

                for i in range(2):
                    agt = ag.create_group(f'Trial {i}')
                    agt.attrs.create('Sampling rate', 50.0)

                    agt.create_dataset('Accelerometer', data=np.random.rand(1000, 3) + np.array([[0, 0, 1]]))

    yield [study1_path, study2_path]

    # clean up the temporary directories
    study1_td.cleanup()
    study2_td.cleanup()


