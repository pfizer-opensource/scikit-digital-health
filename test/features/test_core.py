import pytest

from skimu.features.core import (
    get_n_feats,
    partial_index_check,
    normalize_indices,
    normalize_axes,
    Bank,
    Feature,
    ArrayConversionError,
)
from skimu.features.lib.moments import Mean, StdDev, Skewness, Kurtosis


@pytest.mark.parametrize(
    ("index", "truth"),
    (
            (1, 1),
            ([0, 2], 2),
            (slice(0, 3, 2), 2),
            (..., 3)
    )
)
def test_get_n_feats(index, truth):
    pred = get_n_feats(3, index)

    assert pred == truth


def test_partial_index_check():
    with pytest.raises(IndexError):
        partial_index_check("test")
    with pytest.raises(IndexError):
        partial_index_check(13513.0)
    with pytest.raises(IndexError):
        partial_index_check(get_n_feats)

    assert isinstance(partial_index_check(None), type(...))
    assert partial_index_check([0, 2]) == [0, 2]


class TestNormalizeIndices:
    def test_none(self):
        assert normalize_indices(3, None) == [..., ..., ...]

    def test_int(self):
        assert normalize_indices(3, 1) == [1, 1, 1]

    def test_iterable(self):
        assert normalize_indices(3, [0, 2]) == [[0, 2], [0, 2], [0, 2]]

    def test_sequence(self):
        res = normalize_indices(2, [[0, 2], 1])
        assert res == [[0, 2], 1]

    def test_slice(self):
        assert normalize_indices(2, slice(0, 3, 2)) == [slice(0, 3, 2), slice(0, 3, 2)]

    def test_function_input_error(self):
        with pytest.raises(IndexError):
            normalize_indices(3, get_n_feats)


class TestNormalizeAxes:
    def test_1d(self):
        assert normalize_axes(1, 0, 500) == (0, None)

    @pytest.mark.parametrize(
        ("ndim", "ax", "idx_ax", "truth"),
        (
                (2, 0, -1, (0, 0)),
                (2, 0, None, (0, None)),
                (3, 0, 1, (0, 0)),
                (3, 1, None, (1, None)),
                (3, -1, 1, (2, 1)),
                (4, 1, 3, (1, 2)),
                (4, 2, 3, (2, 2))
        )
    )
    def test_nd(self, ndim, ax, idx_ax, truth):
        assert normalize_axes(ndim, ax, idx_ax) == truth

    def test_same_axis(self):
        with pytest.raises(ValueError):
            normalize_axes(3, 1, 1)


class TestBank:
    def setup(self):
        pass

    def test_dunder_methods(self):
        bank = Bank()
        bank.add([Mean(), StdDev(), Skewness()])

        # contains
        assert Mean() in bank
        assert StdDev() in bank
        assert Skewness() in bank
        assert Kurtosis() not in bank

        # length
        assert len(bank) == 3

    def test_add(self):
        bank = Bank()

        bank.add(Mean(), [0, 2])

        assert bank._feats == [Mean()]
        assert bank._indices == [[0, 2]]

        bank.add([StdDev(), Skewness()], [[0, 2], 1])

        assert bank._feats == [Mean(), StdDev(), Skewness()]
        assert bank._indices == [[0, 2], [0, 2], 1]

        with pytest.warns(UserWarning):
            bank.add(Mean())
        with pytest.warns(UserWarning):
            bank.add([Skewness(), Kurtosis()])

    def test_save(self, temp_bank_file):
        bank = Bank()
        bank.add([Mean(), StdDev(), Skewness()], [..., [0, 2], 1])

        bank.save(temp_bank_file)

    def test_load(self, temp_bank_file):
        bank = Bank()
        bank.load(temp_bank_file)

        assert bank._feats == [Mean(), StdDev(), Skewness()]
        assert bank._indices == [..., [0, 2], 1]

        bank2 = Bank(temp_bank_file)

        assert bank2._feats == [Mean(), StdDev(), Skewness()]
        assert bank2._indices == [..., [0, 2], 1]
