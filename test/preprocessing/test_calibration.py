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
            array([1.00816559, 1.00257388, 1.00343983])
        )  # GGIR: array([1.00731463792368, 1.00268720163593, 1.00349275392835])
        assert allclose(
            cal_res["offset"],
            array([-0.00029996, -0.00026138, -0.00280073])
        )  # GGIR: array([-0.000182315190654408, -0.000315021317133069, -0.00273379237669694])
        assert allclose(
            cal_res["temperature scale"],
            array([[-5.62854755e-05, -4.41293910e-06,  3.01197144e-04]])
        )  # GGIR: array([-4.13744070488941e-05, 8.93468717108127e-06, 0.000296623190202781])
