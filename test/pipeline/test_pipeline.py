import pytest

from numpy import allclose

from ..base_conftest import resolve_data_path, get_truth_data

from skimu import Pipeline
from skimu.pipeline import NotAProcessError
from skimu.gait import Gait
from skimu.sit2stand import Sit2Stand
from skimu.read import ReadCWA


class TestPipeline:
    gait_keys = [
        'IC',
        'FC',
        'delta h',
        'b valid cycle',
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

    def test(self):
        p = Pipeline()

        p.add(ReadCWA(base=None, period=None))
        p.add(Gait(
            use_cwt_scale_relation=True,
            min_bout_time=8.0,
            max_bout_separation_time=0.5,
            max_stride_time=2.25,
            loading_factor=0.2,
            height_factor=0.53,
            prov_leg_length=False,
            filter_order=4,
            filter_cutoff=20.0
        ))

        file = resolve_data_path('ax3_sample.cwa', 'pipeline')

        res = p.run(file=file, height=1.88)

        # get the truth data
        f = get_truth_data()
        gait_res = f(
            resolve_data_path('gait_data.h5', 'pipeline'),
            self.gait_keys
        )

        for key in gait_res:
            assert allclose(res['Gait Process'][key], gait_res[key], equal_nan=True), \
                f'{key} does not match truth'

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