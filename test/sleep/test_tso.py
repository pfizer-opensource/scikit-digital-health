import numpy as np

from skimu.sleep.tso import get_total_sleep_opportunity


class TestDetectTSO:
    def test(self, tso_dummy_data):
        data, sleep = tso_dummy_data(20.0)
        time, acc, temp, lux = data

        # calculate tso
        tso = get_total_sleep_opportunity(
            20., time, acc, temp,
            np.array([0]), np.array([time.size]),
            min_rest_block=30,
            max_act_break=60,
            min_angle_thresh=0.1,
            max_angle_thresh=1.0,
            int_wear_temp=25.0,
            int_wear_move=0.001,
            plot_fn=lambda x: None,
            idx_start=0
        )

        assert abs(tso[0] - sleep[0]) < 30  # less than 30 seconds off
        assert abs(tso[1] - sleep[1]) < 30
