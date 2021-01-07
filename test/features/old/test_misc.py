import inspect

import pytest
from numpy import random

from skimu.features import *
from skimu.features import lib
from skimu.features.core2 import _normalize_axes


class TestNormalizeAxes:
    @pytest.mark.parametrize(
        ("ndim", "ax", "ca", "true_ax", "true_ca"),
        (
                (1, -1, -2, 0, -1),
                (2, -1, -2, 1, 0),
                (3, -1, -2, 2, 1),
                (3, 0, 1, 0, 1),
                (3, 0, 2, 0, 0),
                (3, 1, 2, 1, 1),
                (3, 2, 1, 2, 1)
        )
    )
    def test_nd_dims(self, ndim, ax, ca, true_ax, true_ca):
        pax, pca = _normalize_axes(ndim, False, ax, ca)

        assert pax == true_ax
        assert pca == true_ca

    @pytest.mark.parametrize(
        ("ax", "ca", "true_ax", "true_ca"),
        (
                (-1, -2, 0, 0),
                (-2, -1, 0, 0),
                (0, 1, 0, 0),
                (1, 0, 0, 0)
        )
    )
    def test_df_dims(self, ax, ca, true_ax, true_ca):
        pax, pca = _normalize_axes(2, True, ax, ca)

        assert pax == true_ax
        assert pca == true_ca

    def test_same_error(self):
        with pytest.raises(ValueError):
            _normalize_axes(8, False, 3, 3)

        with pytest.raises(ValueError):
            _normalize_axes(8, False, -1, -1)

        with pytest.raises(ValueError):
            _normalize_axes(2, True, 0, 0)


class TestFeatureClass:
    def test_indexing_errors(self):
        with pytest.raises(ValueError):
            mn = Mean()[[1, ...]]

        with pytest.raises(ValueError):
            mn = Mean()[:5, :10]

        with pytest.raises(ValueError):
            mn = Mean()[[1, ...]]

        with pytest.raises(ValueError):
            mn = Mean()['test']

    def test_get_columns(self):
        res = Mean()._get_columns(['x'])

        assert res == ['Mean_x']


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
