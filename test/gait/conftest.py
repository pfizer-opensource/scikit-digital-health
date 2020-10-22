from tempfile import TemporaryDirectory as TempDir
from pathlib import Path

from pytest import fixture
import h5py
import numpy as np
from scipy.interpolate import interp1d

from ..base_conftest import *


@fixture(scope='module')
def sample_accel():
    path = resolve_data_path('gait_data.h5', 'gait')
    with h5py.File(path, 'r') as f:
        accel = f['accel'][()]

    return accel


@fixture(scope='module')
def sample_dt():
    path = resolve_data_path('gait_data.h5', 'gait')
    with h5py.File(path, 'r') as f:
        dt = np.mean(np.diff(f['time'][:500]))

    return dt


@fixture(scope='module')
def sample_time():
    path = resolve_data_path('gait_data.h5', 'gait')
    with h5py.File(path, 'r') as f:
        accel = f['time'][()]

    return accel


@fixture(scope='module')
def get_sample_bout_accel():
    def get_stuff(freq):
        with h5py.File(resolve_data_path('gait_data.h5', 'gait'), 'r') as f:
            accel = f['accel'][()]
            fs = 1 / np.mean(np.diff(f['time']))

        if freq >= 50.0:
            with h5py.File(resolve_data_path('gait_data.h5', 'gait'), 'r') as f:
                bout = f['Truth']['Gait Classification']['gait_classification_50'].attrs.get('bout')

            bout_acc = accel[bout[0]:bout[1], :]
        else:
            with h5py.File(resolve_data_path('gait_data.h5', 'gait'), 'r') as f:
                bout = f['Truth']['Gait Classification']['gait_classification_50'].attrs.get('bout')

            f = interp1d(
                np.arange(0, accel.shape[0] / fs, 1 / fs),
                accel,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate',
                axis=0
            )
            acc_ds = f(np.arange(0, accel.shape[0] / fs, 1 / 20.0))

            bout_acc = acc_ds[bout[0]:bout[1], :]

        vaxis = np.argmax(np.mean(bout_acc, axis=0))
        return bout_acc, vaxis, np.sign(np.mean(bout_acc, axis=0)[vaxis])

    return get_stuff


@fixture(scope='module')
def get_contact_truth():
    def get_stuff(bout=0):
        with h5py.File(resolve_data_path('gait_data.h5', 'gait'), 'r') as f:
            mask = f['Truth']['Bout N'][()] == bout
            ic = f['Truth']['IC'][mask]
            fc = f['Truth']['FC'][mask]

        return ic, fc
    return get_stuff


@fixture(scope='module')
def get_gait_classification_truth():
    def get_stuff(freq):
        if freq >= 50.0:
            with h5py.File(resolve_data_path('gait_data.h5', 'gait'), 'r') as f:
                truth = f['Truth']['Gait Classification']['gait_classification_50'][()]
        else:
            with h5py.File(resolve_data_path('gait_data.h5', 'gait'), 'r') as f:
                truth = f['Truth']['Gait Classification']['gait_classification_20'][()]

        return truth
    return get_stuff


@fixture(scope='module')
def get_bgait_samples_truth():  # boolean gait classification
    def get_stuff(case):
        bgait = np.zeros(1000, dtype=np.bool_)

        bouts_ = [
            (0, 90),
            (150, 160),
            (165, 180),
            (200, 210),
            (225, 240),
            (400, 760),
            (770, 780),
            (990, 1000)
        ]
        for bout in bouts_:
            bgait[bout[0]:bout[1]] = True

        if case == 1:
            dt = 1 / 50
            n_max_sep = 25  # 0.5 seconds
            n_min_time = 75  # 1.5 seconds

            bouts = [
                (0, 90),
                (150, 240),
                (400, 780)
            ]
        elif case == 2:
            dt = 1 / 100
            n_max_sep = 50  # 0.5 seconds
            n_min_time = 200  # 2 seconds

            bouts = [
                (400, 780)
            ]

        elif case == 3:
            dt = 1 / 50
            n_max_sep = 75  # 1.5 seconds
            n_min_time = 5  # 0.1 seconds

            bouts = [
                (0, 240),
                (400, 780),
                (990, 1000)
            ]
        else:
            dt = 1 / 50
            n_max_sep = 6  # 0.12 seconds
            n_min_time = 5  # 0.1 seconds

            bouts = [
                (0, 90),
                (150, 180),
                (200, 210),
                (225, 240),
                (400, 760),
                (770, 780),
                (990, 1000)
            ]
        return bgait, dt, n_max_sep * dt, n_min_time * dt, bouts
    return get_stuff


@fixture
def sample_gait():
    gait = {
        'IC': np.tile(np.array([10, 35, 62, 86, 111]), 2),
        'FC opp foot': np.tile(np.array([15, 41, 68, 90, 116]), 2),
        'FC': np.tile(np.array([40, 65, 90, 115, 140]), 2),
        'delta h': np.tile(np.array([0.05, 0.055, 0.05, 0.045, np.nan]), 2),
        'Bout N': np.repeat([1, 2], 5)
    }
    return gait


@fixture(scope='module')
def sample_gait_aux():
    def y(x):
        return np.sin(np.pi * x / x.max()) + np.sin(5 * np.pi * x/x.max()) / (x+1)

    a = np.concatenate(
        (
            y(np.arange(25)),
            y(np.arange(27)),
            y(np.arange(24)),
            y(np.arange(25)),
            y(np.arange(25)),
            y(np.arange(24)),
            y(np.arange(24)),
            y(np.arange(24)),
            y(np.arange(24)),
            y(np.arange(25)),
            y(np.arange(27))
        )
    ).reshape((-1, 1))

    gait_aux = {
        'accel': [
            a, a
        ],
        'inertial data i': np.repeat([0, 1], 5),
        'vert axis': 0
    }

    return gait_aux


@fixture(scope='class')
def sample_datasets():
    study1_td = TempDir()
    study1_path = Path(study1_td.name)
    study2_td = TempDir()
    study2_path = Path(study2_td.name)

    for k in range(2):
        # study 1
        with h5py.File(study1_path / f'subject_{k}.h5', 'w') as f:
            for j in range(3):
                ag = f.create_group(f'activity{j}')
                ag.attrs.create('Gait Label', 1 if j == 1 else 0)

                for i in range(2):
                    agt = ag.create_group(f'Trial {i}')
                    agt.attrs.create('Sampling rate', 100.0)

                    agt.create_dataset('Accelerometer', data=np.random.rand(1500, 3) + np.array([[0, 0, 1]]))

    # study 2
        with h5py.File(study2_path / f'subject_{k}.h5', 'w') as f:
            for j in range(2):
                ag = f.create_group(f'activity{j}')
                ag.attrs.create('Gait Label', 1 if j == 1 else 0)

                for i in range(3):
                    agt = ag.create_group(f'Trial {i}')
                    agt.attrs.create('Sampling rate', 50.0)

                    agt.create_dataset('Accelerometer', data=np.random.rand(1000, 3) + np.array([[0, 1, 0]]))

    yield [study1_path, study2_path]

    # clean up the temporary directories
    study1_td.cleanup()
    study2_td.cleanup()
