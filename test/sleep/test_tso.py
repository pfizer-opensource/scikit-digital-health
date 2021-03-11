from skimu.sleep.tso import detect_tso


class TestDetectTSO:
    def test(self, tso_dummy_data):
        data, sleep = tso_dummy_data(20.0)
        time, acc, temp, lux = data

        # calculate tso
        tso = detect_tso(
            acc, time, 20.0, temp, min_rest_block=30, allowed_rest_break=60,
            min_angle_threshold=0.1, max_angle_threshold=1.0, move_td=0.001, temp_td=25.0
        )

        assert abs(tso[0] - sleep[0]) < 30  # less than 30 seconds off
        assert abs(tso[1] - sleep[1]) < 30
