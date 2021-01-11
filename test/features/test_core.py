"""
Testing core functionality for the feature module
"""
from tempfile import NamedTemporaryFile

import pytest
from numpy import allclose
from numpy.random import random
from pandas import DataFrame

from skimu.features import *
from skimu.features import lib


class TestFeatureBank:
    def test_add(self):
        bank = Bank()

        bank.add([
            Mean(),
            Range(),
            StdDev()
        ])

        assert bank._feats == [Mean(), Range(), StdDev()]

        return bank

    @pytest.mark.parametrize(
        ("index_", "index_length"),
        (
                (0, 1),
                (slice(0, 10, 2), 5),  # 0, 2, 4, 6, 8
                ([0, 4], 2),
                (slice(1, 7), 6),  # 1, 2, 3, 4, 5, 6
                (..., 10)
        )
    )
    def test_add_single_index(self, index_, index_length):
        bank = Bank()

        bank.add([Mean(), Range(), StdDev()], index=index_)

        assert all([i == bank._indices[0] for i in bank._indices])

        x = random((10, 100, 150))
        res = bank.compute(x, fs=20., axis=-1, index_axis=0)

        assert res.shape == (len(bank) * index_length, 100)

    @pytest.mark.parametrize(
        ("index_", "index_count", "index_equal"),
        (
                ([3, [0, 2], slice(0, 10, 2), ...], 18, False),
                ([[3], [4], [0], [1]], 4, False),  # 1 index per feature
                ([[3], 4, 0, 1], 4, False),  # this should also be 1 per feature
                ([3, 4, 0, 1], 16, True),  # this should be applied to each feature, 4*4
                (..., 40, True),
                (4, 4, True),
                (None, 40, True)
        )
    )
    def test_add_multiple_index(self, index_, index_count, index_equal):
        bank = Bank()

        bank.add([Mean(), Range(), RMS(), IQR()], index=index_)

        if index_equal:
            assert all([i == bank._indices[0] for i in bank._indices])

        bank.add(StdDev())  # another 10 elements, default is to include all elements

        x = random((10, 100, 150))
        res = bank.compute(x, fs=20., axis=-1, index_axis=0)

        assert res.shape == (index_count + 10, 100)

    def test_columns(self):
        bank = self.test_add()

        x = DataFrame(data=random((100, 3)), columns=['x', 'y', 'z'])
        res = bank.compute(x, axis=0, index_axis=1, columns=['x', 'z'])

        assert res.shape == (2 * 3,)

    """
    |  shape       | axis  | ind_ax |  res shape   |
    |--------------|-------|--------|--------------|
    | (a, b)       |   0   |    1   | (bf,)        |
    | (a, b)       |   0   |    N   | (f, b)       |
    | (a, b)       |   1   |    0   | (3a,)        |
    | (a, b)       |   1   |    N   | (f, a)       |
    | (a, b, c)    |   0   |  1(0)  | (bf, c)      |
    | (a, b, c)    |   0   |  2(1)  | (b, cf)      |
    | (a, b, c)    |   0   |  N     | (f, b, c)    |
    | (a, b, c)    |   1   |  0     | (af, c)      |
    | (a, b, c)    |   1   |  2(1)  | (a, cf)      |
    | (a, b, c)    |   1   |  N     | (f, a, c)    |
    | (a, b, c)    |   2   |  0     | (af, b)      |
    | (a, b, c)    |   2   |  1     | (a, bf)      |
    | (a, b, c)    |   2   |  N     | (f, a, b)    |
    | (a, b, c, d) |   0   |  1(0)  | (bf, c, d)   |
    | (a, b, c, d) |   0   |  2(1)  | (b, cf, d)   |
    | (a, b, c, d) |   0   |  3(2)  | (d, c, df)   |
    | (a, b, c, d) |   0   |  N     | (f, b, c, d) |
    | (a, b, c, d) |   1   |  0     | (af, c, d)   |
    | (a, b, c, d) |   1   |  2(1)  | (a, cf, d)   |
    | (a, b, c, d) |   1   |  3(2)  | (a, c, df)   |
    | (a, b, c, d) |   1   |  N     | (f, a, c, d) |
    | (a, b, c, d) |   2   |  0     | (af, b, d)   |
    | (a, b, c, d) |   2   |  1     | (a, bf, d)   |
    | (a, b, c, d) |   2   |  3(2)  | (a, b, df)   |
    | (a, b, c, d) |   2   |  N     | (f, a, b, d) |
    | (a, b, c, d) |   3   |  0     | (af, b, c)   |
    | (a, b, c, d) |   3   |  1     | (a, bf, c)   |
    | (a, b, c, d) |   3   |  2     | (a, b, cf)   |
    | (a, b, c, d) |   3   |  N     | (f, a, b, c) |
    """
    @pytest.mark.parametrize(
        ("in_shape", "axis", "caxis", "out_shape"),
        (
                # 1D
                (150, 0, None, (3,)),
                # 2D
                ((5, 10), 0, 1, (10 * 3,)),
                ((5, 10), 0, None, (3, 10)),
                ((5, 10), 1, 0, (5 * 3,)),
                ((5, 10), 1, None, (3, 5)),
                # 3D
                ((5, 10, 15), 0, 1, (10*3, 15)),
                ((5, 10, 15), 0, 2, (10, 15*3)),
                ((5, 10, 15), 0, None, (3, 10, 15)),
                ((5, 10, 15), 1, 0, (5*3, 15)),
                ((5, 10, 15), 1, 2, (5, 15*3)),
                ((5, 10, 15), 1, None, (3, 5, 15)),
                ((5, 10, 15), 2, 0, (5*3, 10)),
                ((5, 10, 15), 2, 1, (5, 10*3)),
                ((5, 10, 15), 2, None, (3, 5, 10)),
                # some of 4D
                ((5, 10, 15, 20), 0, 2, (10, 15*3, 20)),
                ((5, 10, 15, 20), 0, None, (3, 10, 15, 20)),
                ((5, 10, 15, 20), 2, 0, (5*3, 10, 20))
        )
    )
    def test_shape(self, in_shape, axis, caxis, out_shape):
        bank = self.test_add()
        x = random(in_shape)

        res = bank.compute(x, 20., axis=axis, index_axis=caxis)

        assert res.shape == out_shape

    def test_shape_df(self):
        bank = self.test_add()
        x = DataFrame(data=random((100, 5)))

        res = bank.compute(x)
        assert res.shape == (15,)

    def test_contains(self):
        bank = self.test_add()

        bank.add(DominantFrequencyValue(padlevel=4, low_cutoff=0., high_cutoff=5.))

        assert Mean() in bank
        assert DominantFrequency() not in bank
        assert DominantFrequency not in bank

        assert DominantFrequencyValue(padlevel=4, low_cutoff=0., high_cutoff=5.) in bank
        assert DominantFrequencyValue(padlevel=2, low_cutoff=0., high_cutoff=5.) not in bank

    def test_save_load(self):
        bank = self.test_add()

        bank_file = NamedTemporaryFile('r+')

        x = random((5, 100, 150))
        truth1 = bank.compute(x, fs=20., axis=-1, index_axis=None)
        truth2 = bank.compute(x, fs=20., axis=-1, index_axis=0)

        bank.save(bank_file.name)

        bank2 = Bank(bank_file=bank_file.name)
        res1 = bank2.compute(x, fs=20., axis=-1, index_axis=None)
        res2 = bank2.compute(x, fs=20., axis=-1, index_axis=0)

        assert allclose(res1, truth1)
        assert allclose(res2, truth2)

        bank3 = Bank()
        bank3.load(bank_file.name)

        bank_file.close()

        res3 = bank3.compute(x, fs=20., axis=-1, index_axis=None)
        res4 = bank3.compute(x, fs=20., axis=-1, index_axis=0)

        assert allclose(res3, truth1)
        assert allclose(res4, truth2)


class TestFeature:
    def test_equivalence(self):
        f1_a = DominantFrequency(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)
        f1_b = DominantFrequency(padlevel=4, low_cutoff=0.5, high_cutoff=9.0)
        f2 = SPARC(padlevel=4, fc=10., amplitude_threshold=0.05)
        f3 = Range()

        assert f1_a == f1_a
        assert f1_b == f1_b
        assert f2 == f2
        assert f3 == f3

        assert f1_a != f1_b
        assert f1_a != f2
        assert f1_a != f3
        assert f2 != f3

    def test_output_shape(self):
        for ndim in range(5):
            for axis in range(ndim - 1):
                in_shape = (5, 10, 15, 20, 25)[:ndim+1]
                out_shape = tuple(s for i, s in enumerate(in_shape) if i != axis)  # remove the axis of computation
                x = random(in_shape)
                for ft in lib.__all__:
                    f = getattr(lib, ft)()  # default parameters

                    # if testing detail power stuff, need more values
                    if 'Detail' in ft:
                        shape = list(in_shape)
                        shape[axis] = 150
                        x = random(tuple(shape))

                    res = f.compute(x, axis=axis, fs=50.)

                    assert res.shape == out_shape, f"Failed on feature {f!s} for ndim {ndim} " \
                                                   f"and axis {axis}. Input " \
                                                   f"shape {in_shape} -> {out_shape}"
