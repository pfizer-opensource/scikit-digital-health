import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

from numpy import isclose, isnan

from skdh.io import ReadCSV


class TestHandleTimestampInconsistency:
    def test_block_timestamps(self, dummy_csv_contents):
        # save for multiple uses
        kw = dict(
            fill_dict={"accel": 1.0, "temperature": 0.0},
        )
        # get the fixture contents
        raw, fs, n_full = dummy_csv_contents(drop=True)

        # setup reader to be able to test timestamp handling
        rdr = ReadCSV(
            "time",
            {"accel": ["ax", "ay", "az"], "temperature": ["temperature"]},
            fill_gaps=True,
        )

        # get numpy arrays
        time = raw["_datetime_"].astype(int).values / 1e9
        data = {
            "accel": raw[["ax", "ay", "az"]].values,
            "temperature": raw["temperature"].values,
        }

        comp_fs, time_rs, data_rs = rdr.handle_timestamp_inconsistency_np(
            **kw, time=time, data=data
        )

        assert isclose(comp_fs, fs)
        assert time_rs.size == n_full
        # check that the nans were filled
        for _, dstream in data_rs.items():
            assert dstream.shape[0] == n_full
            assert isnan(dstream).sum() == 0

        # trim a few samples off the last block and check we get a warning
        time2 = raw["_datetime_"].astype(int).values[:-5] / 1e9
        data2 = {
            "accel": raw[["ax", "ay", "az"]].values[:-5],
            "temperature": raw["temperature"].values[:-5],
        }
        with pytest.warns(UserWarning, match="Non integer number of blocks"):
            comp_fs2, time2_rs, data2_rs = rdr.handle_timestamp_inconsistency_np(
                **kw, time=time2, data=data2
            )

        assert isclose(comp_fs2, fs)
        assert time2_rs.size == int(n_full - fs)
        for _, dstream in data2_rs.items():
            assert dstream.shape[0] == int(n_full - fs)

    def test_unequal_blocks(self, dummy_csv_contents):
        kw = dict(
            fill_dict={"accel": 1.0, "temperature": 0.0},
        )
        rdr = ReadCSV(
            "time",
            {"accel": ["ax", "ay", "az"], "temperature": ["temperature"]},
            fill_gaps=True,
        )

        raw, fs, n_full = dummy_csv_contents(drop=True)

        # drop a few blocks to create uneven blocks of timestamps
        raw.drop(index=range(713, 723), inplace=True)
        raw.drop(index=range(13095, 14003), inplace=True)
        raw.reset_index(drop=True, inplace=True)

        time = raw["_datetime_"].astype(int).values / 1e9
        data = {
            "accel": raw[["ax", "ay", "az"]].values,
            "temperature": raw["temperature"].values,
        }

        with pytest.raises(ValueError, match="not all equal size"):
            rdr.handle_timestamp_inconsistency_np(**kw, time=time, data=data)

    def test_reader(self, dummy_csv_contents):
        raw, fs, n_full = dummy_csv_contents(drop=True)

        rdr = ReadCSV(
            time_col_name="_datetime_",
            column_names={"accel": ["ax", "ay", "az"], "temperature": "temperature"},
            raw_conversions={"accel": 1.0},
        )
        # read with only temperature data, but we still have accel column names
        rdr2 = ReadCSV(
            time_col_name="_datetime_",
            column_names={"accel": ["ax", "ay", "az"], "temperature": "temperature"},
            read_csv_kwargs={"usecols": (4, 5)},
        )

        # send raw data to io for reading
        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "test.csv"
            raw.to_csv(fname, index=False)

            res = rdr.predict(file=fname)
            with pytest.warns():
                res2 = rdr2.predict(file=fname)

        assert res["time"].size == n_full
        assert res["fs"] == fs
        assert res["temperature"].ndim == 1
        assert res["accel"].ndim == 2
        assert res["accel"].shape == (n_full, 3)

        assert "accel" not in res2
        assert "time" in res2
        assert "temperature" in res2
