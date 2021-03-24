import numpy as np

from skimu.utility.internal import rle


class TestRLE:
    def test_full_expected_input(self):
        arr = [0] * 5 + [1] * 3 + [0] * 4 + [1] * 7 + [0] * 2 + [1] * 6 + [0] * 1
        exp_lengths = np.asarray([5, 3, 4, 7, 2, 6, 1])
        exp_indices = np.asarray([0, 5, 8, 12, 19, 21, 27])
        exp_values = np.asarray([0, 1, 0, 1, 0, 1, 0])

        lengths, starts, vals = rle(arr)

        assert np.allclose(lengths, exp_lengths)
        assert np.allclose(starts, exp_indices)
        assert np.allclose(vals, exp_values)

    def test_single_val(self):
        arr = [0] * 50
        exp_lengths = np.asarray([50])
        exp_indices = np.asarray([0])
        exp_values = np.asarray([0])

        lengths, starts, vals = rle(arr)

        assert np.allclose(lengths, exp_lengths)
        assert np.allclose(starts, exp_indices)
        assert np.allclose(vals, exp_values)