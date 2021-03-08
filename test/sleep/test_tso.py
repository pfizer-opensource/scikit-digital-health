from numpy import repeat
from pandas import to_datetime

from skimu.sleep.tso import detect_tso


def check_tso_detection():
    from skimu.read import ReadBin

    src = "/Users/ladmin/Desktop/PfyMU_development/sleeppy_pfymu/test_data/demo.bin"
    reader = ReadBin(base=12, period=24)
    res = reader.predict(src)
    for day in res["day_ends"]:
        acc = res["accel"][day[0] : day[1]]
        t = res["time"][day[0] : day[1]]
        fs = 100
        temp = res["temperature"][day[0] : day[1]]
        temp = repeat(temp, 300)
        out = detect_tso(acc=acc, t=t, fs=fs, temp=temp)
    print(out)
    print(to_datetime(out[0], unit="s"), to_datetime(out[1], unit="s"))
