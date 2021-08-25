import numpy as np

from skdh.utility import fragmentation_endpoints as fe


class TestGini:
    """
    Test values pulled from here:
    https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-
    python/48999797#48999797
    """

    def test1(self):
        x = np.array([1, 1, 1, 1, 1000])

        assert np.isclose(np.around(fe.gini(x, corr=False), 3), 0.796)
        assert np.isclose(np.around(fe.gini(x, corr=True), 3), 0.995)

    def test2(self):
        x = np.array([3, 1, 6, 2, 1])
        w = np.array([4, 2, 2, 10, 1])

        assert np.isclose(np.around(fe.gini(x, w=w, corr=False), 4), 0.2553)
        assert np.isclose(
            np.around(fe.gini(x, w=w, corr=True), 4), np.around(0.2553 * 5 / 4, 4)
        )