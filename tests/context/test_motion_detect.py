import pytest

from skdh.context import MotionDetectionAlgorithm


class TestMotionDetection:
    def test_motion_positive(self, motion_positive_data):
        acc, fs = motion_positive_data
        mda = MotionDetectionAlgorithm()
        res = mda.predict(accel=acc, fs=fs)
        assert res["movement_detected"].all()

    def test_motion_negative(self, motion_negative_data):
        acc, fs = motion_negative_data
        mda = MotionDetectionAlgorithm()
        res = mda.predict(accel=acc, fs=fs)
        assert (~res["movement_detected"]).any()
