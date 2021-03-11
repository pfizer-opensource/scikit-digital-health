"""
Testing of utility functions for sleep analysis.

Yiorgos Christakis
Pfizer DMTI 2021
"""
import pytest
import numpy as np
from skimu.sleep.utility import *


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
        arr = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0])
        out = arg_longest_bout(arr, 1)

        expected = 1, 4
        assert out == expected
