import pytest

from skimu.features.core2 import Feature

from .conftest import F1, F2, F3


class TestBaseFeature:
    def test_abc(self):
        with pytest.raises(TypeError):
            Feature()

    def test_equivalence(self):
        f1_a = F1(p2=10, p1=5)
        f1_b = F1(p1=5, p2=10)[0]
        f2 = F2(p1=5, p2=10)
        f3 = F3()

        assert f1_a == f1_a
        assert f1_b == f1_b
        assert f1_a != f1_b
        assert f1_a != f2
        assert f1_b != f2
        assert f2 != f3

    def test_indexing(self):
        f = F1()

        f[0]
        assert f.n == 1
        assert f.index == 0

        f[:]
        assert f.n == -1
        assert f.index == slice(None)

        f[[0, 2]]
        assert f.n == 2
        assert f.index == [0, 2]

        f[(0, 1)]
        assert f.n == 2
        assert f.index == (0, 1)

        f[slice(2)]
        assert f.n == -1  # should be -1 since possible to not use the whole slice
        assert f.index == slice(2)

    def test_indexing_errors(self):
        f = F1()
        error = ValueError
        with pytest.raises(error):
            f[..., 2]

        with pytest.raises(error):
            f[..., [0, 2]]

        with pytest.raises(error):
            f[:, 0]

        with pytest.raises(error):
            f[:, :, 0]

        with pytest.raises(error):
            f[:, :, [0, 1]]

        with pytest.raises(error):
            f[:, :]
