import numpy as np

from skdh.sleep.tso import get_total_sleep_opportunity


class TestDetectTSO:
    def test(self, tso_dummy_data):
        data, sleep = tso_dummy_data(20.0)
        time, acc, temp, lux = data

        # calculate tso
        tso = get_total_sleep_opportunity(
            20.0,
            time,
            acc,
            temp,
            np.array([0]),
            np.array([time.size]),
            min_rest_block=30,
            max_act_break=60,
            tso_min_thresh=0.1,
            tso_max_thresh=1.0,
            tso_perc=10,
            tso_factor=15.0,
            int_wear_temp=25.0,
            int_wear_move=0.001,
            plot_fn=lambda x: None,
            idx_start=0,
        )

        assert abs(tso[0] - sleep[0]) < 30  # less than 30 seconds off
        assert abs(tso[1] - sleep[1]) < 30

    def test_all_moving(self):
        x = np.zeros((50000, 3))
        x[:, 2] = np.random.default_rng(seed=5).random(x.shape[0]) + 1
        temp = np.zeros(x.shape[0]) + 18.0
        time = np.arange(1.6e9, 1.6e9 + x.shape[0] / 20, 1 / 20)

        tso = get_total_sleep_opportunity(
            20.0,
            time,
            x,
            temp,
            np.array([0]),
            np.array([time.size]),
            min_rest_block=30,
            max_act_break=60,
            tso_min_thresh=0.1,
            tso_max_thresh=1.0,
            tso_perc=10,
            tso_factor=15.0,
            int_wear_temp=25.0,
            int_wear_move=0.001,
            plot_fn=lambda y: None,
            idx_start=0,
        )

        assert tso == (None, None, None, None)
