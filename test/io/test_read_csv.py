import pytest
from numpy import isclose

from skdh.io import ReadCSV
from skdh.io.csv import handle_timestamp_inconsistency, handle_accel, handle_windows


class TestHandleTimestampInconsistency:
    def test_block_timestamps(self, dummy_csv_contents):
        # save for multiple uses
        kw = dict(fill_gaps=True, accel_col_names=['ax', 'ay', 'az'], accel_in_g=True, g=9.81)
        # get the fixture contents
        raw, fs, n_full = dummy_csv_contents

        df_full, comp_fs = handle_timestamp_inconsistency(raw.copy(), **kw)

        assert isclose(comp_fs, fs)
        assert df_full.shape[0] == n_full
        # check that the nans were filled
        assert df_full['ax'].isna().sum() == 0
        assert df_full['ay'].isna().sum() == 0
        assert df_full['az'].isna().sum() == 0

        # trim a few samples off the last block and check we get a warning
        raw2 = raw.iloc[:-5]
        with pytest.warns(UserWarning, match="Non integer number of blocks"):
            df_full2, comp_fs2 = handle_timestamp_inconsistency(raw2.copy(), **kw)

        assert isclose(comp_fs2, fs)
        assert df_full2.shape[0] == int(n_full - fs)
