"""
Test core sleep module functionality

Yiorgos Christakis, Lukas Adamowicz
Pfizer DMTI 2019-2021
"""
from skimu import Pipeline
from skimu.read import ReadBin
from skimu.sleep import Sleep


class TestSleep:
    def test(self):
        file = "/Users/lukasadamowicz/Documents/Packages/sleeppy/sleeppy/test/test_data/demo.bin"
        # file = "/Users/lukasadamowicz/Downloads/STEPP_QC/0074_GNACTV_LeftWrist.bin"
        file = "/Users/lukasadamowicz/Downloads/SQUAD_acc_0000_nonsensitive_20200101_100706184018_left_2.bin"

        slp = Sleep(
            start_buffer=0,
            stop_buffer=0,
            min_rest_block=30,
            max_activity_break=150,
            min_angle_thresh=0.1,
            max_angle_thresh=1.0,
            min_rest_period=None,
            nonwear_move_thresh=None,
            min_wear_time=0,
            min_day_hours=6,
            downsample=True,
            day_window=(12, 24)
        )
        slp.setup_plotting("test.pdf")

        p = Pipeline()
        p.add(ReadBin(bases=[12], periods=[24]))
        p.add(
            slp,
            save_results=True,
            save_name="sleep_results.csv"
        )

        p.run(file=file)

        assert True
