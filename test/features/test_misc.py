import inspect

import pytest
from numpy import random

from skimu.features import *
from skimu.features import lib
from skimu.features.core import DeferredFeature, NotAFeatureError


class TestFeatureEquivalence:
    @pytest.mark.parametrize(
        ('f1', 'f2'),
        (
                (Mean(), Mean()),
                (DominantFrequency(low_cutoff=0.0, high_cutoff=12.0),
                 DominantFrequency(low_cutoff=0.0, high_cutoff=12.0)),
                (JerkMetric(normalize=True), JerkMetric(normalize=True)),
                (Kurtosis()['x', 'y'], Kurtosis()['z'])
        )
    )
    def test(self, f1, f2):
        assert f1 == f2

    @pytest.mark.parametrize('f', lib.__all__)
    def test_not_equal(self, f):
        feature = getattr(lib, f)

        argspec = inspect.getfullargspec(feature)

        kwargs = self.get_kwargs(argspec, feature)
        f1 = feature(**kwargs)

        # test with non feature item
        assert all([f1 != i for i in [list(), float(), tuple(), dict()]])

        if len(argspec.args) <= 1:
            pytest.skip('No parameters to test feature difference')
        for i, arg in enumerate(argspec.args[1:]):
            if argspec.defaults[i] is None:
                continue
            else:
                f2 = feature(**self.change_kwarg(kwargs, arg, feature))

                assert f1 != f2
                assert all([f2 != i for i in [list(), float(), tuple(), dict()]])

    @pytest.mark.parametrize('idx', (1.1, 'a', [1.1, 'a'], (1.5, 2.5)))
    def test_index_error(self, idx):
        with pytest.raises(IndexError):
            m = Mean()[idx]

    @staticmethod
    def get_kwargs(spec, feat):

        kw = {}

        for i, arg in enumerate(spec.args[1:]):
            kw[arg] = TestFeatureEquivalence.rand(
                spec.defaults[i],
                arg,
                feat
            )

        return kw

    @staticmethod
    def rand(like_type, arg, feat):
        if isinstance(like_type, int):
            return random.randint(0, 10)
        elif isinstance(like_type, float):
            return random.rand()
        elif isinstance(like_type, bool):
            return bool(random.randint(0, 2))
        elif isinstance(like_type, str):
            options = getattr(feat, f'_{arg}_options')
            return options[random.randint(0, len(options))]
        elif like_type is None:
            return None
        else:
            raise ValueError(f"can't deal with type {type(like_type)} for {feat!r}")

    @staticmethod
    def change_kwarg(kw, key, feat):
        kwargs = kw.copy()

        new = kwargs[key]
        while new == kwargs[key] and new is not None:
            new = TestFeatureEquivalence.rand(kwargs[key], key, feat)

        kwargs[key] = new

        return kwargs


class TestDeferredFeature:
    @pytest.mark.parametrize('f', lib.__all__)
    def test_equivalence(self, f):
        feature = getattr(lib, f)

        df1 = DeferredFeature(feature(), ...)
        df2 = DeferredFeature(feature(), ...)
        f1 = feature()

        assert df1 == df2
        assert df1 == f1

    @pytest.mark.parametrize(
        ('f1', 'f2'),
        (
                (Mean(), JerkMetric()),
                (DominantFrequency(), DetailPower())
        )
    )
    def test_not_equal(self, f1, f2):
        df1 = DeferredFeature(f1, ...)
        df2 = DeferredFeature(f2, ...)

        assert df1 != df2
        assert df1 != f2
        assert f1 != df2
        # misc other testing
        assert df1 != list()
        assert df1 != dict()
        assert df2 != list()
        assert df2 != float()

    def test_comp_index(self):
        m = Mean()['xy', 'yz']

        assert m.index == [0, 2]

    def test_get_columns(self):
        feat = DeferredFeature(Mean(), ...)

        columns = feat.get_columns(['x', 'y'])

        assert columns[0] == 'x_Mean()'
        assert columns[1] == 'y_Mean()'

    def test_parent_error(self):
        with pytest.raises(NotAFeatureError):
            DeferredFeature(list(), ...)
