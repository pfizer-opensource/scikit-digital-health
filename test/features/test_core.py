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
