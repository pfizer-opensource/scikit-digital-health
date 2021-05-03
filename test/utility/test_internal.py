import numpy as np

from skimu.utility.internal import rle, get_day_index_intersection


class TestGetDayIndexIntersection:
    day_start = 2000
    day_stop = 4000

    # treat sleep as exclusionary
    sleep_starts = {
        1: np.array([200, 1200, 2200, 4200]),
        2: np.array([200, 1800, 3800]),
        3: np.array([200, 1500, 4200])
    }
    sleep_stops = {
        1: np.array([800, 1800, 2800, 5000]),
        2: np.array([800, 2500, 4400]),
        3: np.array([200, 1900, 5000])
    }

    wear_starts = np.array([0, 2300, 3000])
    wear_stops = np.array([1800, 2900, 3900])

    # solutions
    starts = {
        1: np.array([2800, 3000]),
        2: np.array([2500, 3000]),
        3: np.array([2300, 3000])
    }
    stops = {
        1: np.array([2900, 3900]),
        2: np.array([2900, 3800]),
        3: np.array([2900, 3900])
    }

    so_starts = {  # sleep only
        1: np.array([2000, 2800]),
        2: np.array([2500]),
        3: np.array([2000])
    }
    so_stops = {
        1: np.array([2200, 4000]),
        2: np.array([3800]),
        3: np.array([4000])
    }

    def test(self):
        for i in range(1, 4):
            p_starts, p_stops = get_day_index_intersection(
                (self.sleep_starts[i], self.wear_starts),
                (self.sleep_stops[i], self.wear_stops),
                (False, True),
                self.day_start,
                self.day_stop
            )

            assert np.allclose(p_starts, self.starts[i])
            assert np.allclose(p_stops, self.stops[i])

    def test_sleep_only(self):
        for i in range(1, 4):
            p_starts, p_stops = get_day_index_intersection(
                self.sleep_starts[i],
                self.sleep_stops[i],
                False,
                self.day_start,
                self.day_stop
            )

            assert np.allclose(p_starts, self.so_starts[i])
            assert np.allclose(p_stops, self.so_stops[i])


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