import pytest
import numpy as np

from skdh.context import MotionDetectRCoV, MotionDetectRCoVMaxSD


class TestMotionDetectRCoV:
    # Test algorithm on data with 100% movement
    def test_motion_positive(self, motion_positive_data_100hz):
        time, acc, fs = motion_positive_data_100hz
        mda = MotionDetectRCoV()
        res = mda.predict(time=time, accel=acc, fs=fs)
        assert res["movement_detected"].all()

    # Test algorithm on data with 0% movement
    def test_motion_negative(self, motion_negative_data_100hz):
        time, acc, fs = motion_negative_data_100hz
        mda = MotionDetectRCoV()
        res = mda.predict(time=time, accel=acc, fs=fs)
        assert (~res["movement_detected"]).all()

    # Test input check runs on real data
    def test_input_check(self, motion_positive_data_20hz):
        time, accel, _ = motion_positive_data_20hz
        mda = MotionDetectRCoV()
        assert len(mda._check_input(time, accel)) == 3

    # Test input check row requirement
    def test_input_check_rows(self):
        with pytest.raises(ValueError) as e_info:
            MotionDetectRCoV()._check_input(
                time=np.arange(0, 19 / 20, 1 / 20), accel=np.ones([19, 3])
            )

    # Test input check column requirement
    def test_input_check_columns(self):
        with pytest.raises(ValueError) as e_info:
            MotionDetectRCoV()._check_input(
                time=np.arange(0, 60 / 20, 1 / 20), accel=np.ones([60, 2])
            )

    # Test input check sample rate requirement
    def test_input_check_fs(self):
        with pytest.raises(ValueError) as e_info:
            MotionDetectRCoV()._check_input(
                time=np.arange(0, 120 / 10, 1 / 10), accel=np.ones([120, 3])
            )


class TestMotionDetectRCoVMaxSD:
    # Test algorithm on data with 100% movement (20hz)
    def test_motion_positive_20(self, motion_positive_data_20hz):
        time, acc, fs = motion_positive_data_20hz
        mda = MotionDetectRCoVMaxSD()
        res = mda.predict(time=time, accel=acc, fs=fs)
        assert res["movement_detected"].all()

    # Test algorithm on data with 0% movement (20hz)
    def test_motion_negative_20(self, motion_negative_data_20hz):
        time, acc, fs = motion_negative_data_20hz
        mda = MotionDetectRCoVMaxSD()
        res = mda.predict(time=time, accel=acc, fs=fs)
        assert (~res["movement_detected"]).all()

    # Test algorithm on data with 100% movement (100hz)
    def test_motion_positive_100(self, motion_positive_data_100hz):
        time, acc, fs = motion_positive_data_100hz
        mda = MotionDetectRCoVMaxSD()
        res = mda.predict(time=time, accel=acc, fs=fs)
        assert res["movement_detected"].all()

    # Test algorithm on data with 0% movement (100hz)
    def test_motion_negative_100(self, motion_negative_data_100hz):
        time, acc, fs = motion_negative_data_100hz
        mda = MotionDetectRCoVMaxSD()
        res = mda.predict(time=time, accel=acc, fs=fs)
        assert (~res["movement_detected"]).all()

    # Test input check runs on real data
    def test_input_check(self, motion_positive_data_20hz):
        time, accel, _ = motion_positive_data_20hz
        mda = MotionDetectRCoVMaxSD()
        assert len(mda._check_input(time, accel)) == 3

    # Test input check row requirement
    def test_input_check_rows(self):
        with pytest.raises(ValueError) as e_info:
            MotionDetectRCoVMaxSD()._check_input(
                time=np.arange(0, 19 / 20, 1 / 20), accel=np.ones([19, 3])
            )

    # Test input check column requirement
    def test_input_check_columns(self):
        with pytest.raises(ValueError) as e_info:
            MotionDetectRCoVMaxSD()._check_input(
                time=np.arange(0, 60 / 20, 1 / 20), accel=np.ones([60, 2])
            )

    # Test input check sample rate requirement
    def test_input_check_fs(self):
        with pytest.raises(ValueError) as e_info:
            MotionDetectRCoVMaxSD()._check_input(
                time=np.arange(0, 120 / 10, 1 / 10), accel=np.ones([120, 3])
            )
