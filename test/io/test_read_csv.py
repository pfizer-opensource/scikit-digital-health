import pytest
from numpy import isclose, allclose, array

from skdh.io import ReadCSV
from skdh.io.csv import handle_timestamp_inconsistency, handle_accel, handle_windows


class TestHandleTimestampInconsistency:
    def test_block_timestamps(self, dummy_csv_contents):
        # save for multiple uses
        kw = dict(fill_gaps=True, accel_col_names=['ax', 'ay', 'az'], accel_in_g=True, g=9.81)
        # get the fixture contents
        raw, fs, n_full = dummy_csv_contents(drop=True)

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

    def test_unequal_blocks(self, dummy_csv_contents):
        kw = dict(fill_gaps=True, accel_col_names=['ax', 'ay', 'az'], accel_in_g=True, g=9.81)

        raw, fs, n_full = dummy_csv_contents(drop=True)

        # drop a few blocks to create uneven blocks of timestamps
        raw.drop(index=range(713, 723), inplace=True)
        raw.drop(index=range(13095, 14003), inplace=True)
        raw.drop(index=range(987131, 987139), inplace=True)
        raw.reset_index(drop=True, inplace=True)

        with pytest.raises(ValueError, match="not all equal size"):
            handle_timestamp_inconsistency(raw, **kw)


class TestHandleWindows:
    def test_no_run(self, dummy_csv_contents):
        raw, fs, n_full = dummy_csv_contents()

        time = raw["_datetime_"]

        out = handle_windows(time, [0], [24], run_windowing=False)

        assert out == {}

    def test_single_window(self, dummy_csv_contents):
        raw, fs, n_full = dummy_csv_contents(drop=False)
        time = raw['_datetime_']

        out = handle_windows(time, [14], [3], run_windowing=True)

        truth = array([
            [int(2 * 3600 * fs), (int((2 + 3) * 3600 * fs))],
            [int((24 + 2) * 3600 * fs), int((24 + 2 + 3) * 3600 * fs)],
            [int((48 + 2) * 3600 * fs), int((48 + 2 + 3) * 3600 * fs)]
        ])

        assert allclose(out[(14, 3)], truth)

    def test_multiple_windows(self, dummy_csv_contents):
        raw, fs, n_full = dummy_csv_contents(drop=False)
        time = raw['_datetime_']

        out = handle_windows(time, [14, 10], [3, 8], run_windowing=True)

        truth_14_3 = array([
            [int(2 * 3600 * fs), (int((2 + 3) * 3600 * fs))],
            [int((24 + 2) * 3600 * fs), int((24 + 2 + 3) * 3600 * fs)],
            [int((48 + 2) * 3600 * fs), int((48 + 2 + 3) * 3600 * fs)]
        ])

        truth_10_8 = array([
            [0, int((18 - 12) * 3600 * fs)],
            [int((24 - 2) * 3600 * fs), int((24 + 8 - 2) * 3600 * fs)],
            [int((48 - 2) * 3600 * fs), int((48 + 8 - 2) * 3600 * fs)],
            [int((72 - 2) * 3600 * fs), n_full],
        ])

        assert len(out) == 2
        assert allclose(out[(14, 3)], truth_14_3)
        assert allclose(out[(10, 8)], truth_10_8)
