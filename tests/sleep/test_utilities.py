"""
Testing of utility functions for sleep analysis.

Yiorgos Christakis
Pfizer DMTI 2021
"""

import numpy as np

from skdh.utility import moving_median
from skdh.sleep.utility import *
from skdh.sleep.utility import get_weartime


class TestGetWearTime:
    def test(self):
        fs = 50.0

        rng = np.random.default_rng(seed=10)
        x = rng.normal(loc=[0, 0, 1], scale=2, size=(int(4 * 3600 * fs), 3))
        t = rng.normal(loc=28, scale=0.5, size=x.shape[0])

        n1 = int(1.5 * 3600 * fs)
        n2 = n1 + int(3600 * fs)

        x[n1:n2] = [0, 0, 1]
        t[n1:n2] = 22.0
        rmd = moving_median(x, 250, 1, axis=0)

        wt = get_weartime(rmd, t, fs, 0.001, 25.0)

        assert np.isclose(
            (wt[0][1] - wt[1][0]) / fs, 1800, atol=10
        )  # 10 second tolerance


class TestComputeZAngle:
    def test_x_dir(self):
        acc = np.asarray(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
        )
        out = compute_z_angle(acc)
        expected = np.asarray([0, 0, 0, 0, 0, 0])
        assert np.array_equal(out, expected)

    def test_y_dir(self):
        acc = np.asarray(
            [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        )
        out = compute_z_angle(acc)
        expected = np.asarray([0, 0, 0, 0, 0, 0])
        assert np.array_equal(out, expected)

    def test_z_dir(self):
        acc = np.asarray(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        )
        out = compute_z_angle(acc)
        expected = np.asarray([90, 90, 90, 90, 90, 90])
        assert np.array_equal(out, expected)


class TestComputeAbsoluteDifference:
    def test_uniform_array(self):
        arr = np.ones([10, 1])
        out = compute_absolute_difference(arr)
        expected = np.zeros([10, 1])
        assert np.array_equal(out, expected)

    def test_alternating_array(self):
        arr = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        out = compute_absolute_difference(arr)
        expected = np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(out, expected)

    def test_random_array(self):
        arr = np.asarray([1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        out = compute_absolute_difference(arr)
        expected = np.asarray([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        assert np.array_equal(out, expected)


class TestDropMinBlocks:
    def test_uniform(self):
        arr = np.ones([10, 1])
        out = drop_min_blocks(
            arr, min_block_size=5, drop_value=1, replace_value=0, skip_bounds=False
        )
        expected = np.ones([10, 1])
        assert np.array_equal(out, expected)

    def test_alternating(self):
        arr = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        out = drop_min_blocks(
            arr, min_block_size=5, drop_value=1, replace_value=0, skip_bounds=False
        )
        expected = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(out, expected)

    def test_skip_bounds(self):
        arr = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        out = drop_min_blocks(
            arr, min_block_size=5, drop_value=1, replace_value=0, skip_bounds=True
        )
        expected = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(out, expected)

    def test_random(self):
        arr = np.asarray([1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        out = drop_min_blocks(
            arr, min_block_size=2, drop_value=0, replace_value=2, skip_bounds=False
        )
        expected = np.asarray([1, 1, 2, 1, 0, 0, 0, 0, 1, 1])
        assert np.array_equal(out, expected)

    def test_value_not_present(self):
        arr = np.asarray([1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        out = drop_min_blocks(
            arr, min_block_size=2, drop_value=2, replace_value=3, skip_bounds=False
        )
        expected = np.asarray([1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        assert np.array_equal(out, expected)

    def test_replace(self):
        arr = np.asarray([1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        out = drop_min_blocks(
            arr, min_block_size=3, drop_value=1, replace_value=3, skip_bounds=False
        )
        expected = np.asarray([3, 3, 0, 3, 0, 0, 0, 0, 3, 3])
        assert np.array_equal(out, expected)

    def test_replace_skip(self):
        arr = np.asarray([1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        out = drop_min_blocks(
            arr, min_block_size=3, drop_value=1, replace_value=3, skip_bounds=True
        )
        expected = np.asarray([1, 1, 0, 3, 0, 0, 0, 0, 1, 1])
        assert np.array_equal(out, expected)


class TestArgLongestBout:
    def test(self):
        arr = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0])

        out = arg_longest_bout(arr, 1)
        assert out == (1, 4)

        out = arg_longest_bout(arr, 0)
        assert out == (7, 11)

    def test_with_ends(self):
        arr = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])

        out = arg_longest_bout(arr, 1)
        assert out == (0, 3)

        out = arg_longest_bout(arr, 0)
        assert out == (15, 20)

    def test_2_same_length(self):
        arr = np.array(
            [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]
        )
        out = arg_longest_bout(arr, 1)

        expected = 1, 4
        assert out == expected

    def test_one_value(self):
        arr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        out = arg_longest_bout(arr, 0)

        assert out == (None, None)


class TestComputeActivityIndex:
    def test(self):
        rng = np.random.default_rng(seed=5)
        x = np.arange(120 * 3, dtype=np.float64).reshape((-1, 3))
        x += rng.normal(loc=0.0, scale=0.5, size=x.shape)

        res = compute_activity_index(1.0, x, hp_cut=1e-3)

        assert np.allclose(res, np.array([0.4047359, 0.45388503]))
