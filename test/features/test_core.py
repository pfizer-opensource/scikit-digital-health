"""
Testing core functionality for the feature module
"""
import pytest
from numpy import allclose
from numpy.random import random
from pandas import DataFrame

from skimu.features import *


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

        assert all([i.index == index_ for i in bank])

        x = random((10, 100, 150))
        res = bank.compute(x, fs=20., axis=-1, col_axis=0)

        assert res.shape == (len(bank) * index_length, 100)

    @pytest.mark.parametrize(
        ("index_", "index_count", "index_equal"),
        (
                ([3, [0, 2], slice(0, 10, 2), ...], 18, False),
                ([[3], [4], [0], [1]], 4, False),  # 1 index per feature
                ([[3], 4, 0, 1], 4, False),  # this should also be 1 per feature
                ([3, 4, 0, 1], 16, True),  # this should be applied to each feature, 4*4
                (..., 40, True),
                (4, 4, True)
        )
    )
    def test_add_multiple_index(self, index_, index_count, index_equal):
        bank = Bank()

        bank.add([Mean(), Range(), RMS(), IQR()], index=index_)

        if index_equal:
            assert all([bank._indices[i] == index_ for i in range(4)])

        bank.add(StdDev())  # another 10 elements, default is to include all elements

        x = random((10, 100, 150))
        res = bank.compute(x, fs=20., axis=-1, col_axis=0)

        assert res.shape == (index_count + 10, 100)

    def test_add_index_no_col_axis_error(self):
        bank = Bank()

        bank.add([Mean(), Range()], index=[0, 4])

        x = random((10, 100, 150))

        # TODO change this to specific error
        with pytest.raises(Exception):
            res = bank.compute(x, fs=20., axis=-1, col_axis=None)

    def test_columns(self):
        bank = self.test_add()

        x = random((10, 100, 150))
        res = bank.compute(x, axis=-1, col_axis=0, columns=slice(0, 10, 3))  # 0, 3, 6, 9

        assert res.shape == (4 * 3, 100)

        x = DataFrame(data=random((100, 3)), columns=['x', 'y', 'z'])
        res = bank.compute(x, axis=0, col_axis=1, columns=['x', 'z'])

        assert res.shape == (100, 2 * 3)

    @pytest.mark.parametrize(
        ("in_shape", "axis", "caxis", "out_shape"),
        (
                (150, 0, None, (3,)),
                ((5, 10), 0, 1, (5 * 3,)),
                ((5, 10), 0, None, (3, 5)),
                ((5, 10), 1, 0, (5 * 3,)),
                ((5, 10), 1, None, (3, 5)),
                ((5, 10, 15), 0, 1, (10*3, 15)),
                ((5, 10, 15), 0, 2, (10, 15*3)),
                ((5, 10, 15), 0, None, (3, 10, 15)),
                ((5, 10, 15), 1, 0, (5*3, 15)),
                ((5, 10, 15), 1, 2, (5, 15*3)),
                ((5, 10, 15), 1, None, (3, 5, 15)),
                ((5, 10, 15), 2, 0, (5*3, 10)),
                ((5, 10, 15), 2, 1, (5, 10*3)),
                ((5, 10, 15), 2, None, (3, 5, 10))
        )
    )
    def test_shape(self, in_shape, axis, caxis, out_shape):
        """
        |  shape | ax | ia |  swap1 | ax | ia |  res  | ax | ia | res swap |
        |--------|----|----|--------|----|----|-------|----|----|----------|
        | (a, b) |  0 |  1 | (b, a) |  0 |  0 | (bf,) |    |    |          |
        | (a, b) |  0 |  N | (b, a) |  0 |  N | (f, b)|    |    |          |
        | (a, b) |  1 |  0 |        |    |    | (3a,) |    |    |          |
        | (a, b) |  1 |  N |        |    |    | (f, a)|    |    |          |

        |  shape   | ax| ia|   swap1  | ax| ia|   swap2  | ax | ia |   res    | ia |   | res swap |
        |----------|---|---|----------|---|---|----------|----|----|----------|----|---|----------|
        | (a, b, c)| 0 | 1 | (c, b, a)| 2 | 1 | (b, c, a)|  2 |  0 | (bf, c)  |  0 | 0 | (bf, c)  |
        | (a, b, c)| 0 | 2 | (c, b, a)| 2 | 0 |          |    |    | (cf, b)  |  0 | 1 | (b, cf)  |
        | (a, b, c)| 0 | N | (c, b, a)| 2 | N |          |    |    | (f, c, b)|  1 | 2 | (f, b, c)|
        | (a, b, c)| 1 | 0 | (a, c, b)| 2 | 0 |          |    |    | (af, c)  |  0 | 0 | (af, c)  |
        | (a, b, c)| 1 | 2 | (a, c, b)| 2 | 1 | (c, a, b)|  2 |  0 | (cf, a)  |  0 | 1 | (a, cf)  |
        | (a, b, c)| 1 | N | (a, c, b)| 2 | N |          |    |    | (f, a, c)|  1 | 1 | (f, a, c)|
        | (a, b, c)| 2 | 0 |          |   |   |          |    |    | (af, b)  |  0 | 0 | (af, b)  |
        | (a, b, c)| 2 | 1 |          |   |   | (b, a, c)|  2 |  0 | (bf, a)  |  0 | 1 | (a, bf)  |
        | (a, b, c)| 2 | N |          |   |   |          |    |    | (f, a, b)|  1 | 1 | (f, a, b)|
        """
        bank = self.test_add()
        x = random(in_shape)

        res = bank.compute(x, 20., axis=axis, col_axis=caxis)

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

    def test_save_load(self, bank_file):
        bank = self.test_add()

        x = random((5, 100, 150))
        truth1 = bank.compute(x, fs=20., axis=-1, col_axis=None)
        truth2 = bank.compute(x, fs=20., axis=-1, col_axis=0)

        bank.save(bank_file)

        bank2 = Bank(bank_file=bank_file)
        res1 = bank2.compute(x, fs=20., axis=-1, col_axis=None)
        res2 = bank2.compute(x, fs=20., axis=-1, col_axis=0)

        assert allclose(res1, truth1)
        assert allclose(res2, truth2)

        bank3 = Bank()
        bank3.load(bank_file)

        res3 = bank3.compute(x, fs=20., axis=-1, col_axis=None)
        res4 = bank3.compute(x, fs=20., axis=-1, col_axis=0)

        assert allclose(res3, truth1)
        assert allclose(res4, truth2)

