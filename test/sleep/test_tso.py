from numpy import repeat
from pandas import to_datetime

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

"""
def check_tso_detection():
    from skimu.read import ReadBin

    src = "/Users/ladmin/Desktop/PfyMU_development/sleeppy_pfymu/test_data/demo.bin"
    reader = ReadBin(base=12, period=24)
    res = reader.predict(src)
    for day in res["day_ends"]:
        acc = res["accel"][day[0]: day[1]]
        t = res["time"][day[0]: day[1]]
        fs = 100
        temp = res["temperature"][day[0] : day[1]]
        temp = repeat(temp, 300)
        out = detect_tso(acc=acc, t=t, fs=fs, temp=temp)
    print(out)
    print(to_datetime(out[0], unit="s"), to_datetime(out[1], unit="s"))
"""