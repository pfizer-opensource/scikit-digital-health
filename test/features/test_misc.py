import inspect

import pytest
from numpy import random

from skimu.features import *
from skimu.features import lib


class _TestFeatureEquivalence:
    @pytest.mark.parametrize(
        ('f1', 'f2'),
        (
                (Mean(), Mean()),
                (DominantFrequency(low_cutoff=0.0, high_cutoff=12.0),
                 DominantFrequency(low_cutoff=0.0, high_cutoff=12.0)),
                (JerkMetric(), JerkMetric()),
                (Kurtosis()[[0, 1]], Kurtosis()[0, 1])
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
            kw[arg] = _TestFeatureEquivalence.rand(
                spec.defaults[i],
                arg,
                feat
            )

        return kw

    def rand(self, like_type, arg, feat):
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

    def change_kwarg(self, kw, key, feat):
        kwargs = kw.copy()

        new = kwargs[key]
        while new == kwargs[key] and new is not None:
            new = self.rand(kwargs[key], key, feat)

        kwargs[key] = new

        return kwargs
