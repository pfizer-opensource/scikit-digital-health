"""
Testing of utility functions for sleep analysis.

Yiorgos Christakis
Pfizer DMTI 2021
"""
import pytest
import numpy as np
from src.skimu.sleep.utility import *


class TestRollingMean:
    def test_rolling_step_row_vector(self):
        arr = np.arange(10)
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
        out = rolling_mean(arr, w_size=3, step=1)
        assert np.array_equal(expected, out)

    def test_window_step_row_vector(self):
        arr = np.arange(10)
        expected = np.asarray([1.0, 4.0, 7.0]).reshape(-1, 1)
        out = rolling_mean(arr, w_size=3, step=3)
        assert np.array_equal(expected, out)

    def test_rolling_step_column_vector(self):
        arr = np.arange(10).reshape(-1, 1)
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
        out = rolling_mean(arr, w_size=3, step=1)
        assert np.array_equal(expected, out)

    def test_window_step_column_vector(self):
        arr = np.arange(10).reshape(-1, 1)
        expected = np.asarray([1.0, 4.0, 7.0]).reshape(-1, 1)
        out = rolling_mean(arr, w_size=3, step=3)
        assert np.array_equal(expected, out)


class TestRollingSTD:
    def test_rolling_step_row_vector(self):
        arr = np.arange(10)
        expected = np.ones((8, 1))
        out = rolling_std(arr, w_size=3, step=1)
        assert np.array_equal(expected, out)

    def test_window_step_row_vector(self):
        arr = np.arange(10)
        expected = np.ones((3, 1))
        out = rolling_std(arr, w_size=3, step=3)
        assert np.array_equal(expected, out)

    def test_rolling_step_column_vector(self):
        arr = np.arange(10).reshape(-1, 1)
        expected = np.ones((8, 1))
        out = rolling_std(arr, w_size=3, step=1)
        assert np.array_equal(expected, out)

    def test_window_step_column_vector(self):
        arr = np.arange(10).reshape(-1, 1)
        expected = np.ones((3, 1))
        out = rolling_std(arr, w_size=3, step=3)
        assert np.array_equal(expected, out)


class TestRollingMedian:
    def test_rolling_step_row_vector(self):
        arr = np.arange(10)
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
        out = rolling_median(arr, w_size=3, step=1)
        assert np.array_equal(expected, out)

    def test_window_step_row_vector(self):
        arr = np.arange(10)
        expected = np.asarray([1.0, 4.0, 7.0]).reshape(-1, 1)
        out = rolling_median(arr, w_size=3, step=3)
        assert np.array_equal(expected, out)

    def test_rolling_step_column_vector(self):
        arr = np.arange(10).reshape(-1, 1)
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
        out = rolling_median(arr, w_size=3, step=1)
        assert np.array_equal(expected, out)

    def test_window_step_column_vector(self):
        arr = np.arange(10).reshape(-1, 1)
        expected = np.asarray([1.0, 4.0, 7.0]).reshape(-1, 1)
        out = rolling_median(arr, w_size=3, step=3)
        assert np.array_equal(expected, out)


class TestRLE:
    def test_full_expected_input(self):
        arr = [0] * 5 + [1] * 3 + [0] * 4 + [1] * 7 + [0] * 2 + [1] * 6 + [0] * 1
        lengths = np.asarray([5, 3, 4, 7, 2, 6, 1])
        indices = np.asarray([0, 5, 8, 12, 19, 21, 27])
        values = np.asarray([0, 1, 0, 1, 0, 1, 0])
        expected = (lengths, indices, values)
        out = rle(arr)
        assert np.array_equal(expected, out)

    def test_single_val(self):
        arr = [0] * 50
        lengths = np.asarray([50])
        indices = np.asarray([0])
        values = np.asarray([0])
        expected = (lengths, indices, values)
        out = rle(arr)
        assert np.array_equal(expected, out)


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
    def test_uniform(self):
        assert False

    def test_alternating(self):
        assert False

    def test_random(self):
        assert False

    def test_value_not_present(self):
        assert False