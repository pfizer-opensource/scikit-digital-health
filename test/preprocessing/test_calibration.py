"""
Testing Calibration process
"""
from numpy import allclose, array

from skimu.read import ReadBin
from skimu.preprocessing import AccelerometerCalibrate


class TestCalibration:
    def test(self):
        # load the data
        rdr = ReadBin(base=0, period=24)
        res = rdr.predict("/Users/lukasadamowicz/Downloads/long_sample_gnactv.bin")

        # Calibration
        cal = AccelerometerCalibrate()
        cal_res = cal.predict(**res, apply=False)

        assert allclose(
            cal_res["scale"],
            array([1.00674657, 1.0045704, 1.00275762])
        )  # GGIR: array([1.00737395892893, 1.00263241611272, 1.0037194561088])
        assert allclose(
            cal_res["offset"],
            array([0., 0., 0.])
        )  # GGIR: array([-0.000232617106510901, -0.000345580657164915, -0.00219418805647307])
        assert allclose(
            cal_res["temperature scale"],
            array([[-8.34097289e-05, 1.96478451e-04, 5.76586285e-04]])
        )  # GGIR: array([-3.97114439262484e-05, 1.32336733488032e-05, 0.000226993295244372])
