import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

from numpy import isclose

from skdh.io import ReadCSV
from skdh.io.csv import handle_timestamp_inconsistency


class TestHandleTimestampInconsistency:
    def test_block_timestamps(self, dummy_csv_contents):
        # save for multiple uses
        kw = dict(
            fill_gaps=True,
            column_names={'accel': ["ax", "ay", "az"], 'temperature': ['temperature']},
            fill_dict={'accel': 1.0, 'temperature': 0.0},
        )
        # get the fixture contents
        raw, fs, n_full = dummy_csv_contents(drop=True)

        df_full, comp_fs = handle_timestamp_inconsistency(raw.copy(), **kw)

        assert isclose(comp_fs, fs)
        assert df_full.shape[0] == n_full
        # check that the nans were filled
        assert df_full["ax"].isna().sum() == 0
        assert df_full["ay"].isna().sum() == 0
        assert df_full["az"].isna().sum() == 0

        # trim a few samples off the last block and check we get a warning
        raw2 = raw.iloc[:-5]
        with pytest.warns(UserWarning, match="Non integer number of blocks"):
            df_full2, comp_fs2 = handle_timestamp_inconsistency(raw2.copy(), **kw)

        assert isclose(comp_fs2, fs)
        assert df_full2.shape[0] == int(n_full - fs)

    def test_unequal_blocks(self, dummy_csv_contents):
        kw = dict(
            fill_gaps=True,
            column_names={'accel': ["ax", "ay", "az"], 'temperature': ['temperature']},
            fill_dict={'accel': 1.0, 'temperature': 0.0},
        )

        raw, fs, n_full = dummy_csv_contents(drop=True)

        # drop a few blocks to create uneven blocks of timestamps
        raw.drop(index=range(713, 723), inplace=True)
        raw.drop(index=range(13095, 14003), inplace=True)
        raw.reset_index(drop=True, inplace=True)

        with pytest.raises(ValueError, match="not all equal size"):
            handle_timestamp_inconsistency(raw, **kw)
    
    def test_reader(self, dummy_csv_contents):
        raw, fs, n_full = dummy_csv_contents(drop=True)

        rdr = ReadCSV(
            time_col_name="_datetime_",
            column_names={'accel': ['ax', 'ay', 'az'], 'temperature': 'temperature'},
            accel_in_g=True,
        )
        # read with only temperature data, but we still have accel column names
        rdr2 = ReadCSV(
            time_col_name="_datetime_",
            column_names={'accel': ['ax', 'ay', 'az'], 'temperature': 'temperature'},
            accel_in_g=True,
            read_csv_kwargs={'usecols': (4, 5)}
        )

        # send raw data to io for reading
        with TemporaryDirectory() as tdir:
            fname = Path(tdir) / "test.csv"
            raw.to_csv(fname, index=False)

            res = rdr.predict(file=fname)
            with pytest.warns():
                res2 = rdr2.predict(file=fname)

        assert res['time'].size == n_full
        assert res['fs'] == fs
        assert res['temperature'].ndim == 1
        assert res['accel'].ndim == 2
        assert res['accel'].shape == (n_full, 3)

        assert 'accel' not in res2
        assert 'time' in res2
        assert 'temperature' in res2
