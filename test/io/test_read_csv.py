import pytest
from numpy import isclose, allclose, array

from skdh.io import ReadCSV
from skdh.io.csv import handle_timestamp_inconsistency, handle_accel


class TestHandleTimestampInconsistency:
    def test_block_timestamps(self, dummy_csv_contents):
        # save for multiple uses
        kw = dict(
            fill_gaps=True, accel_col_names=["ax", "ay", "az"], accel_in_g=True, g=9.81
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
            fill_gaps=True, accel_col_names=["ax", "ay", "az"], accel_in_g=True, g=9.81
        )

        raw, fs, n_full = dummy_csv_contents(drop=True)

        # drop a few blocks to create uneven blocks of timestamps
        raw.drop(index=range(713, 723), inplace=True)
        raw.drop(index=range(13095, 14003), inplace=True)
        raw.drop(index=range(987131, 987139), inplace=True)
        raw.reset_index(drop=True, inplace=True)

        with pytest.raises(ValueError, match="not all equal size"):
            handle_timestamp_inconsistency(raw, **kw)
