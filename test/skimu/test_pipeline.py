import pytest
from numpy import allclose
from pandas import read_csv

from ..base_conftest import resolve_data_path, get_truth_data

from skimu import Pipeline
from skimu.pipeline import NotAProcessError
from skimu.gait import Gait
from skimu.sit2stand import Sit2Stand
from skimu.read import ReadCWA


class TestPipeline:
    gait_keys = [
        'delta h',
        'PARAM:stride time',
        'PARAM:stance time',
        'PARAM:swing time',
        'PARAM:step time',
        'PARAM:initial double support',
        'PARAM:terminal double support',
        'PARAM:double support',
        'PARAM:single support',
        'PARAM:step length',
        'PARAM:stride length',
        'PARAM:gait speed',
        'PARAM:cadence',
        'PARAM:intra-step covariance - V',
        'PARAM:intra-stride covariance - V',
        'PARAM:harmonic ratio - V',
        'PARAM:stride SPARC',
        'BOUTPARAM:phase coordination index',
        'BOUTPARAM:gait symmetry index',
        'BOUTPARAM:step regularity - V',
        'BOUTPARAM:stride regularity - V',
        'BOUTPARAM:autocovariance symmetry - V',
        'BOUTPARAM:regularity index - V'
    ]

    # some parameters need higher tolerances due to slightly different accelerations
    # some timestamp rounding causes slight changes in the filter cutoffs, effecting the
    # acceleration values
    rtol = {
        'delta h': 2e-3,
        'PARAM:step length': 8.5e-4,
        'PARAM:stride length': 6.5e-4,
        'PARAM:gait speed': 6.5e-4,
        'PARAM:intra-step covariance - V': 2e-3,
        'PARAM:intra-stride covariance - V': 8e-4,
        'PARAM:harmonic ratio - V': 3e-3,
        'PARAM:stride SPARC': 9e-4,
        'BOUTPARAM:gait symmetry index': 2e-5,
        'BOUTPARAM:step regularity - V': 7e-5,
        'BOUTPARAM:stride regularity - V': 1.1e-4,
        'BOUTPARAM:autocovariance symmetry - V': 1.1e-3,
        'BOUTPARAM:regularity index - V': 9e-5
    }

    @staticmethod
    def run_pipeline(get_truth_data, pipe, gait_keys, rel_tol, gait_results_file):
        file = resolve_data_path('ax3_sample.cwa', 'skimu')

        res = pipe.run(file=file, height=1.88)

        # get the truth data
        gait_res = get_truth_data(
            resolve_data_path('gait_data.h5', 'skimu'),
            gait_keys
        )

        for key in gait_res:
            assert allclose(
                res['Gait'][key],
                gait_res[key],
                equal_nan=True,
                rtol=rel_tol.get(key, 1e-8)
            ), f'{key} does not match truth'

        # get the data from the saved file
        data = read_csv(gait_results_file)

        for key in gait_res:
            assert allclose(
                data[key].values,
                gait_res[key],
                equal_nan=True,
                rtol=rel_tol.get(key, 1e-8)
            ), f'{key} from saved data does not match truth'

    def test(self, get_truth_data, gait_res_file, pipeline_file):
        p = Pipeline()

        p.add(ReadCWA(base=None, period=None))
        p.add(
            Gait(
                correct_accel_orient=False,
                use_cwt_scale_relation=True,
                min_bout_time=8.0,
                max_bout_separation_time=0.5,
                max_stride_time=2.25,
                loading_factor=0.2,
                height_factor=0.53,
                prov_leg_length=False,
                filter_order=4,
                filter_cutoff=20.0
            ),
            save_results=True,
            save_name=gait_res_file
        )

        # test saving the pipeline
        p.save(pipeline_file)

        self.run_pipeline(get_truth_data, p, self.gait_keys, self.rtol, gait_res_file)

    def test_load_and_process(self, get_truth_data, gait_res_file, pipeline_file):
        p = Pipeline()
        p.load(pipeline_file)

        self.run_pipeline(get_truth_data, p, self.gait_keys, self.rtol, gait_res_file)

    def test_save_load_not_process_error(self, pipe_file2):
        p = Pipeline()
        p.add(ReadCWA())

        # manually add something wrong
        class NotAProcess:
            pass
        nap = NotAProcess()
        nap.pipe_save = False
        nap.pipe_fname = 'test'
        nap.plot_fname = None

        nap._kw = {'a': 5}
        nap._name = 'NotAProcess'
        nap.__class__.__module__ = 'skimu.notamodule'

        p._steps.append(nap)

        p.save(pipe_file2)

        p2 = Pipeline()

        with pytest.warns(UserWarning):
            p2.load(pipe_file2)

    @pytest.mark.parametrize('proc', (ReadCWA, Gait, Sit2Stand))
    def test_add(self, proc):
        p = Pipeline()

        p.add(proc())

    @pytest.mark.parametrize('not_proc', ([], (), {}, 5.5, 4, 's', None))
    def test_add_error(self, not_proc):
        p = Pipeline()

        with pytest.raises(NotAProcessError):
            p.add(not_proc)

    def test_iteration(self):
        p = Pipeline()

        # override the _steps parameter for first round of testing
        p._steps = list(range(10))

        for i, n in enumerate(p):
            assert n == i

        p._steps = []

        p.add(ReadCWA())
        p.add(Gait())
        p.add(Sit2Stand())

        assert isinstance(next(p), ReadCWA)
        assert isinstance(next(p), Gait)
        assert isinstance(next(p), Sit2Stand)
        with pytest.raises(StopIteration):
            next(p)
