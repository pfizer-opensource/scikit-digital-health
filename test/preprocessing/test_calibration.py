"""
Testing Calibration process
"""
from numpy import allclose, array

from skimu.read import ReadBin
from skimu.preprocessing import Calibrate


class TestCalibration:
    def test(self):
        # load the data
        rdr = ReadBin(base=0, period=24)
        res = rdr.predict("/Users/lukasadamowicz/Downloads/long_sample_gnactv.bin")

        # Calibration
        cal = Calibrate()
        cal_res = cal.predict(**res, apply=False)

        assert allclose(
            cal_res["scale"], array([1.0046647040235, 1.00793883307642, 1.00364200992225])
        )
        assert allclose(
            cal_res["offset"],
            array([-2.95404211756847e-06, 0.000408304543497598, 0.000967419781707366])
        )
        assert allclose(
            cal_res["temp offset"],
            array([-7.08335714072745e-05, 0.000412920869666477, -6.15277802921463e-05])
        )
