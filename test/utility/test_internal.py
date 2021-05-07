import pytest
import numpy as np

from skimu.utility.internal import rle, get_day_index_intersection


class TestGetDayIndexIntersection:
    def test(self, day_ends, sleep_ends, wear_ends, true_intersect_ends):
        day_start, day_stop = day_ends
        sleep_starts, sleep_stops = sleep_ends
        wear_starts, wear_stops = wear_ends
        true_starts, true_stops = true_intersect_ends

        for i in range(1, 4):
            p_starts, p_stops = get_day_index_intersection(
                (sleep_starts[i], wear_starts),
                (sleep_stops[i], wear_stops),
                (False, True),
                day_start,
                day_stop,
            )

            assert np.allclose(p_starts, true_starts[i])
            assert np.allclose(p_stops, true_stops[i])

    def test_sleep_only(self, day_ends, sleep_ends, true_sleep_only_ends):
        day_start, day_stop = day_ends
        sleep_starts, sleep_stops = sleep_ends
        true_starts, true_stops = true_sleep_only_ends

        for i in range(1, 4):
            p_starts, p_stops = get_day_index_intersection(
                sleep_starts[i], sleep_stops[i], False, day_start, day_stop
            )

            assert np.allclose(p_starts, true_starts[i])
            assert np.allclose(p_stops, true_stops[i])

    def test_wear_only(self, day_ends, wear_ends):
        day_start, day_stop = day_ends
        wear_starts, wear_stops = wear_ends

        p_starts, p_stops = get_day_index_intersection(
            wear_starts, wear_stops, True, day_start, day_stop
        )

        assert np.allclose(p_starts, wear_starts[1:])
        assert np.allclose(p_stops, wear_stops[1:])

    def test_mismatch_length_error(self, day_ends, wear_ends):
        day_start, day_stop = day_ends
        wear_starts, wear_stops = wear_ends

        with pytest.raises(ValueError):
            p_starts, p_stops = get_day_index_intersection(
                (wear_starts, np.array([1, 2, 3])),
                wear_stops,
                True,
                day_start,
                day_stop,
            )


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
