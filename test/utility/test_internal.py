import pytest
from numpy import allclose, array, arange

from skimu.utility.internal import get_day_index_intersection, apply_downsample, rle


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
                day_stop
            )

            assert allclose(p_starts, true_starts[i])
            assert allclose(p_stops, true_stops[i])

    def test_sleep_only(self, day_ends, sleep_ends, true_sleep_only_ends):
        day_start, day_stop = day_ends
        sleep_starts, sleep_stops = sleep_ends
        true_starts, true_stops = true_sleep_only_ends

        for i in range(1, 4):
            p_starts, p_stops = get_day_index_intersection(
                sleep_starts[i], sleep_stops[i], False, day_start, day_stop
            )

            assert allclose(p_starts, true_starts[i])
            assert allclose(p_stops, true_stops[i])

    def test_wear_only(self, day_ends, wear_ends):
        day_start, day_stop = day_ends
        wear_starts, wear_stops = wear_ends

        p_starts, p_stops = get_day_index_intersection(
            wear_starts, wear_stops, True, day_start, day_stop
        )

        assert allclose(p_starts, wear_starts[1:])
        assert allclose(p_stops, wear_stops[1:])

    def test_mismatch_length_error(self, day_ends, wear_ends):
        day_start, day_stop = day_ends
        wear_starts, wear_stops = wear_ends

        with pytest.raises(ValueError):
            get_day_index_intersection(
                (wear_starts, array([1, 2, 3])),
                wear_stops,
                True,
                day_start,
                day_stop,
            )


class TestApplyDownsample:
    def test(self, dummy_time, dummy_acc, dummy_idx, dummy_idx_ds):
        tds, (acc_ds,), (idx_ds,) = apply_downsample(
            10.,
            dummy_time,
            (dummy_acc,),
            (dummy_idx,)
        )

        assert allclose(tds, arange(0, 10, 0.1))
        assert acc_ds.shape == (100, 3)
        assert allclose(idx_ds, dummy_idx_ds)

    def test_3d_error(self, dummy_time, np_rng):
        x = np_rng.random((500, 3, 2))

        with pytest.raises(ValueError):
            apply_downsample(10., dummy_time, (x,))


class TestRLE:
    def test_full_expected_input(self, rle_arr, rle_truth):
        pred = rle(rle_arr)

        for p, t in zip(pred, rle_truth):
            assert allclose(p, t)

    def test_single_val(self):
        arr = [0] * 50

        lengths, starts, vals = rle(arr)

        assert allclose(lengths, [50])
        assert allclose(starts, [0])
        assert allclose(vals, [0])
