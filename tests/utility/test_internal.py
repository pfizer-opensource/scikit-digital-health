import pytest
from numpy import allclose, array, arange, random as np_random, concatenate

from skdh.utility.internal import (
    get_day_index_intersection,
    apply_resample,
    rle,
    invert_indices,
    fill_data_gaps,
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

            assert p_starts.size == true_starts[i].size
            assert p_stops.size == true_stops[i].size

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

            assert p_starts.size == true_starts[i].size
            assert p_stops.size == true_stops[i].size

            assert allclose(p_starts, true_starts[i])
            assert allclose(p_stops, true_stops[i])

    def test_include(self):
        day_start, day_stop = 0, 240

        sleep_starts = array([-1, 200, 675, 1165, -1])
        sleep_stops = array([-1, 400, 880, 1360, -1])

        true_starts = array([200])
        true_stops = array([240])

        p_starts, p_stops = get_day_index_intersection(
            sleep_starts, sleep_stops, True, day_start, day_stop
        )

        assert p_starts.size == true_starts.size
        assert p_stops.size == true_stops.size

        assert allclose(p_starts, true_starts)
        assert allclose(p_stops, true_stops)

    def test_include_2(self):
        day_start, day_stop = 200, 400

        starts = array([50, 5, 115, 215, 395, 380, 450, 600])
        stops = array([100, 110, 230, 220, 410, 397, 500, 700])

        true_starts = array([200, 380])
        true_stops = array([230, 400])

        # FOR NOW, this will raise an error with unsupported behavior

        with pytest.raises(NotImplementedError):
            p_starts, p_stops = get_day_index_intersection(
                starts, stops, True, day_start, day_stop
            )

        # assert p_starts.size == true_starts.size
        # assert p_stops.size == true_stops.size
        #
        # assert allclose(p_starts, true_starts)
        # assert allclose(p_stops, true_stops)

    def test_wear_only(self, day_ends, wear_ends):
        day_start, day_stop = day_ends
        wear_starts, wear_stops = wear_ends

        p_starts, p_stops = get_day_index_intersection(
            wear_starts, wear_stops, True, 0, day_stop
        )

        assert p_starts.size == wear_starts.size
        assert p_stops.size == wear_stops.size

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


class TestApplyResample:
    def test_downsample(self, np_rng):
        t = arange(0, 10, 0.1)
        x = np_rng.random((t.size, 3))
        y = np_rng.random((t.size,))

        ix = array([10, 20, 30])
        iy = array([[15, 25], [25, 35]])

        trs, (x_rs, y_rs), (ix_rs, iy_rs) = apply_resample(
            time=t, goal_fs=5.0, data=(x, y), indices=(ix, iy), fs=10.0
        )

        assert allclose(trs, arange(0, 10, 0.2))
        assert x_rs.shape == (50, 3)
        assert y_rs.shape == (50,)
        assert allclose(ix_rs, [5, 10, 15])
        assert allclose(iy_rs, [[7, 12], [12, 18]])

        t_rs = array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        trs, (x_rs, y_rs), (ix_rs, iy_rs) = apply_resample(
            time=t, time_rs=t_rs, data=(x, y), indices=(ix, iy), fs=10.0
        )

        assert allclose(trs, t_rs)
        assert x_rs.shape == (trs.size, 3)
        assert y_rs.shape == (trs.size,)
        assert allclose(ix_rs, [2, 4, 6])
        assert allclose(iy_rs, [[3, 5], [5, 7]])

        t_rs = arange(0, 10, 1 / 3)
        trs, (x_rs, y_rs), (ix_rs, iy_rs) = apply_resample(
            time=t, time_rs=t_rs, data=(x, y), indices=(ix, iy), fs=10.0
        )

        assert allclose(trs, t_rs)
        assert x_rs.shape == (t_rs.size, 3)
        assert y_rs.shape == (t_rs.size,)
        assert allclose(ix_rs, [3, 6, 9])
        assert allclose(iy_rs, [[4, 8], [8, 10]])

    def test_upsample(self, np_rng):
        t = arange(0, 5, 0.5)
        x = arange(t.size)
        ix = array([2, 4, 6])  # 1.0, 2.0, 3.0

        trs, (x_rs,), (ix_rs,) = apply_resample(
            time=t, goal_fs=4.0, data=(x,), indices=(ix,)
        )

        assert trs.size == x_rs.size
        assert allclose(trs, arange(0, 4.5, 0.25))
        assert x_rs.size == 18
        assert ix_rs.size == 3
        assert allclose(x_rs, arange(0, t.size - 1, 0.5))
        assert allclose(ix_rs, [4, 8, 12])


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


class TestFillDataGaps:
    def test(self):
        t = concatenate((arange(0, 4, 0.01), arange(6, 10, 0.01)))
        x = np_random.default_rng().normal(0, 0.5, (t.size, 3))
        temp = np_random.default_rng().normal(28, 0.75, t.size)

        t_rs, data = fill_data_gaps(
            t, 100, {"accel": [0, 0, 1.0]}, accel=x, temperature=temp
        )

        assert "accel" in data
        assert "temperature" in data
        assert allclose(t_rs, arange(0, 10, 0.01))
        assert allclose(data["accel"][400:600], [0.0, 0.0, 1.0])
        assert allclose(data["accel"][:400], x[:400])
        assert allclose(data["accel"][600:], x[400:])
        assert allclose(data["temperature"][400:600], 0.0)

    def test_no_gaps(self):
        t = arange(0, 10, 0.01)
        x = np_random.default_rng().normal(0, 0.5, (t.size, 3))

        t_rs, data = fill_data_gaps(t, 100, {}, accel=x)

        assert "accel" in data
        assert allclose(t_rs, t)
        assert allclose(data["accel"], x)
