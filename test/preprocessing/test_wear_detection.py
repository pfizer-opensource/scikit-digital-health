"""
Unit tests for wear detection algorithms

Lukas Adamowicz
Pfizer DMTI 2021
"""
import numpy as np

from skimu.preprocessing.wear_detection import _modify_wear_times


class TestWearDetection:
    # testing with nonwear on both ends of the data
    def test_modifiction1(self):
        wskip = 15  # minutes
        nh = int(60 / wskip)

        wear_starts_stops = np.array([
            [10, 15],  # 5w surrounded by 10 + 10 nw  : filtered, 5 < 0.3(20)
            [25, 35],  # 10w surrounded by 10 + 1 nw  : kept 10 > 6
            [36, 62],  # 26w surrounded by 1 + 1 nw   : kept 26 > 6
            [63, 67],  # 4 w surrounded by 1 + 6 nw   : kept 4 > 0.3(7)
            [73, 75],  # 2 w surrounded by 5 + 1 nw   : filtered 2 < 0.8(6)
            [76, 77]   # 1 w surrounded by 1 + 4 nw   : filtered 1 < 0.8(5)
        ]) * nh  # convert to blocks of wskip minutes
        """
        After the first pass ->
        starts_stops = [
            [25, 35],  # 10 > 6
            [36, 62],  # 26 > 6
            [63, 67]   # [1][4][14] -> 4 < 0.3(15) removed
        ]
        """

        nonwear = np.ones(81 * nh, dtype=np.bool_)
        for stst in wear_starts_stops:
            nonwear[stst[0]:stst[1]] = False

        # returned starts and stops:
        t_starts = np.array([25, 36]) * nh
        t_stops = np.array([35, 62]) * nh

        starts, stops = _modify_wear_times(nonwear, wskip)

        assert np.allclose(starts, t_starts)
        assert np.allclose(stops, t_stops)

    # testing with non-wear on the end of the data only
    def test_modifiction2(self):
        wskip = 15  # minutes
        nh = int(60 / wskip)

        wear_starts_stops = np.array([
            [0, 15],   # 15w surrounded by 0 + 10 nw  : kept, 15 > 6
            [25, 35],  # 10w surrounded by 10 + 1 nw  : kept 10 > 6
            [36, 62],  # 26w surrounded by 1 + 1 nw   : kept 26 > 6
            [63, 67],  # 4 w surrounded by 1 + 6 nw   : kept 4 > 0.3(7)
            [73, 75],  # 2 w surrounded by 5 + 1 nw   : filtered 2 < 0.8(6)
            [76, 77]   # 1 w surrounded by 1 + 4 nw   : filtered 1 < 0.8(5)
        ]) * nh  # convert to blocks of wskip minutes

        nonwear = np.ones(81 * nh, dtype=np.bool_)
        for stst in wear_starts_stops:
            nonwear[stst[0]:stst[1]] = False

        # returned starts and stops:
        t_starts = np.array([0, 25, 36, 63]) * nh
        t_stops = np.array([15, 35, 62, 67]) * nh

        starts, stops = _modify_wear_times(nonwear, wskip)

        assert np.allclose(starts, t_starts)
        assert np.allclose(stops, t_stops)

    # testing with non-wear at the start of the data
    def test_modifiction3(self):
        wskip = 15  # minutes
        nh = int(60 / wskip)

        wear_starts_stops = np.array([
            [10, 15],  # 5w surrounded by 10 + 10 nw  : filtered, 5 < 0.3(20)
            [25, 35],  # 10w surrounded by 10 + 1 nw  : kept 10 > 6
            [36, 62],  # 26w surrounded by 1 + 1 nw   : kept 26 > 6
            [63, 67],  # 4 w surrounded by 1 + 6 nw   : kept 4 > 0.3(7)
            [73, 75],  # 2 w surrounded by 5 + 1 nw   : filtered 2 < 0.8(6)
            [76, 81]   # 5 w surrounded by 1 + 0 nw   : kept
        ]) * nh  # convert to blocks of wskip minutes

        nonwear = np.ones(81 * nh, dtype=np.bool_)
        for stst in wear_starts_stops:
            nonwear[stst[0]:stst[1]] = False

        # returned starts and stops:
        t_starts = np.array([25, 36, 63, 76]) * nh
        t_stops = np.array([35, 62, 67, 81]) * nh

        starts, stops = _modify_wear_times(nonwear, wskip)

        assert np.allclose(starts, t_starts)
        assert np.allclose(stops, t_stops)

    # testing with wear at both ends
    def test_modifiction4(self):
        wskip = 15  # minutes
        nh = int(60 / wskip)

        wear_starts_stops = np.array([
            [0, 15],   # 15w surrounded by 0 + 10 nw  : kept
            [25, 35],  # 10w surrounded by 10 + 1 nw  : kept 10 > 6
            [36, 62],  # 26w surrounded by 1 + 1 nw   : kept 26 > 6
            [63, 67],  # 4 w surrounded by 1 + 6 nw   : kept 4 > 0.3(7)
            [73, 75],  # 2 w surrounded by 5 + 1 nw   : filtered 2 < 0.8(6)
            [76, 81]   # 5 w surrounded by 1 + 0 nw   : kept
        ]) * nh  # convert to blocks of wskip minutes

        nonwear = np.ones(81 * nh, dtype=np.bool_)
        for stst in wear_starts_stops:
            nonwear[stst[0]:stst[1]] = False

        # returned starts and stops:
        t_starts = np.array([0, 25, 36, 63, 76]) * nh
        t_stops = np.array([15, 35, 62, 67, 81]) * nh

        starts, stops = _modify_wear_times(nonwear, wskip)

        assert np.allclose(starts, t_starts)
        assert np.allclose(stops, t_stops)