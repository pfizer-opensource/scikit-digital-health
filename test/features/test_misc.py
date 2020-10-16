import inspect

import pytest
from numpy import random

from skimu.features import *
from skimu.features import lib


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

        if len(argspec.args) <= 1:
            pytest.skip('No parameters to test feature difference')
        for i, arg in enumerate(argspec.args[1:]):
            if argspec.defaults[i] is None:
                continue
            else:
                f2 = feature(**self.change_kwarg(kwargs, arg, feature))

                assert f1 != f2

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
