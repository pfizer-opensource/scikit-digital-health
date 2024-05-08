from numpy import isclose, allclose

from skdh.io import MultiReader


class TestMultiReader:
    def test_combine(self, dummy_multireader_data):
        _, path_acc, path_eq_temp, path_neq_temp, _ = dummy_multireader_data

        mrdr = MultiReader(
            mode="combine",
            reader="ReadCSV",
            reader_kw={
                "accel": {
                    "time_col_name": "time",
                    "column_names": {"accel": ["ax", "ay", "az"]},
                    "to_datetime_kwargs": {"unit": "s"},
                },
                "temp": {
                    "time_col_name": "time",
                    "column_names": {"temperature": "temperature"},
                    "to_datetime_kwargs": {"unit": "s"},
                },
            },
        )

        res = mrdr.predict(files={"accel": path_acc, "temp": path_eq_temp})

        assert all([i in res for i in ["file", "fs", "time", "accel", "temperature"]])
        assert res["time"].size == res["accel"].shape[0]
        assert res["time"].size == res["temperature"].size
        assert isclose(res["fs"], 20.0)

        # combine with differently sampled timestamps
        res2 = mrdr.predict(files={"accel": path_acc, "temp": path_neq_temp})

        # we are downsampling so should be half the size
        assert all([i in res2 for i in ["file", "fs", "time", "accel", "temperature"]])
        assert res2["time"].size == res2["accel"].shape[0]
        assert res2["time"].size == res2["temperature"].size
        assert isclose(res2["fs"], 10.0)

    def test_concatenate(self, dummy_multireader_data):
        _, path_acc, *_, path_acc_cont = dummy_multireader_data

        mrdr = MultiReader(
            mode="concatenate",
            reader="ReadCSV",
            reader_kw={
                "time_col_name": "time",
                "column_names": {"accel": ["ax", "ay", "az"]},
                "to_datetime_kwargs": {"unit": "s"},
            },
        )

        res = mrdr.predict(files=[path_acc, path_acc_cont])

        assert all([i in res for i in ["file", "fs", "time", "accel"]])
        assert res["time"].size == res["accel"].shape[0]
        assert isclose(res["fs"], 20.0)

        res2 = mrdr.predict(files={"file1": path_acc, "file2": path_acc_cont})

        assert all([i in res2 for i in ["file", "fs", "time", "accel"]])
        assert res2["time"].size == res2["accel"].shape[0]
        assert isclose(res2["fs"], 20.0)

        assert allclose(res2["time"], res["time"])
        assert allclose(res2["accel"], res["accel"])
