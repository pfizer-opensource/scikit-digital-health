import pytest

from skdh.gait_old.get_gait_bouts import get_gait_bouts


@pytest.mark.parametrize("case", (1, 2, 3, 4))
def test_get_gait_bouts(get_bgait_samples_truth, case):
    starts, stops, time, max_sep, min_time, bouts = get_bgait_samples_truth(case)

    pred_bouts = get_gait_bouts(starts, stops, 0, 1500, time, max_sep, min_time)

    assert len(pred_bouts) == len(bouts)
    assert all([b == bouts[i] for i, b in enumerate(pred_bouts)])
