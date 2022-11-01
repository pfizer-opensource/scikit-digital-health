import pytest
from numpy import allclose, array, arange

from skdh.utility.internal import (
    get_day_index_intersection,
    apply_downsample,
    rle,
    invert_indices,
)


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
            wear_starts, wear_stops, True, 0, day_stop
        )

        assert allclose(p_starts, wear_starts)
        assert allclose(p_stops, wear_stops)

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

    def test_empty(self):
        starts, stops = get_day_index_intersection(
            array([]), array([]), (True,), 0, 123456
        )

        # should have no overlapping/intersection
        assert starts.size == 0
        assert stops.size == 0

        # check if false we want the whole time however
        starts, stops = get_day_index_intersection(
            array([]), array([]), (False,), 0, 123456
        )

        # should be full day
        assert starts.size == 1
        assert stops.size == 1
        assert allclose(starts, array([0], dtype=int))
        assert allclose(stops, array([123456], dtype=int))

        # check when we have both empty
        starts, stops = get_day_index_intersection(
            (array([]), array([])), (array([]), array([])), (True, False), 0, 123456
        )

        assert starts.size == 1
        assert stops.size == 1
        assert allclose(starts, array([0], dtype=int))
        assert allclose(stops, array([123456], dtype=int))

    def test_full(self):
        starts, stops = get_day_index_intersection(
            array([0]), array([4000]), (True,), 0, 4000
        )

        assert starts.size == 1
        assert stops.size == 1
        assert starts[0] == 0
        assert stops[0] == 4000


class TestApplyDownsample:
    def test(self, dummy_time, dummy_idx_1d, dummy_idx_2d, np_rng):
        x = np_rng.random((dummy_time.size, 3))
        y = np_rng.random((dummy_time.size,))
        tds, (x_ds, y_ds), (idx_ds_1, idx_ds_2) = apply_downsample(
            10.0, dummy_time, (x, y), (dummy_idx_1d[0], dummy_idx_2d[0])
        )

        assert allclose(tds, arange(0, 10, 0.1))
        assert x_ds.shape == (100, 3)
        assert y_ds.shape == (100,)
        assert allclose(idx_ds_1, dummy_idx_1d[1])
        assert allclose(idx_ds_2, dummy_idx_2d[1])

    def test_none(self, dummy_time):
        tds, (acc_ds,), (idx_ds,) = apply_downsample(10.0, dummy_time, (None,), (None,))

        assert acc_ds is None
        assert idx_ds is None

    def test_3d_error(self, dummy_time, np_rng):
        x = np_rng.random((500, 3, 2))

        with pytest.raises(ValueError):
            apply_downsample(10.0, dummy_time, (x,))


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


class TestInvertIndices:
    def test(self):
        # ||  |-------|       |-------|        |---------|        |----------|   ||
        #  0  50     75      100     110      600       675      800        900  1000
        starts = array([50, 100, 600, 800])
        stops = array([75, 110, 675, 900])

        true_inv_starts = array([0, 75, 110, 675, 900])
        true_inv_stops = array([50, 100, 600, 800, 1000])

        pred_inv_starts, pred_inv_stops = invert_indices(starts, stops, 0, 1000)

        assert allclose(pred_inv_starts, true_inv_starts)
        assert allclose(pred_inv_stops, true_inv_stops)

        # ||-------|       |-------|        |---------|        |----------||
        #  0      75      100     110      600       675      800        900
        starts = array([0, 100, 600, 800])
        stops = array([75, 110, 675, 900])

        true_inv_starts = array([75, 110, 675])
        true_inv_stops = array([100, 600, 800])

        pred_inv_starts, pred_inv_stops = invert_indices(starts, stops, 0, 900)

        assert allclose(pred_inv_starts, true_inv_starts)
        assert allclose(pred_inv_stops, true_inv_stops)

        # ||----------------||
        #  0                900
        starts = array([0])
        stops = array([900])

        pred_inv_starts, pred_inv_stops = invert_indices(starts, stops, 0, 900)

        assert pred_inv_starts.size == 0
        assert pred_inv_stops.size == 0

        # ||       ||
        # 0       900
        pred_inv_starts, pred_inv_stops = invert_indices(array([]), array([]), 0, 900)

        assert allclose(pred_inv_starts, array([0]))
        assert allclose(pred_inv_stops, array([900]))
