from pytest import mark

from src.PfyMU.features import compute_window_samples


class TestLoadDatasets:
    @mark.parametrize(
        ('gfs', 'wlen_s', 'wstep'),
        (
                (30.0, 3.0, 0.5),
                (100.0, 4.0, 0.1)
        )
    )
    def test_2_datasets(self, sample_datasets, gfs, wlen_s, wstep):
        wlen_n, wstep_n = compute_window_samples(gfs, wlen_s, wstep)

        dataset, labels, subjects, activities = load_datasets(
            sample_datasets,
            goal_fs=gfs,
            acc_mag=True,
            window_length=wlen_s,
            window_step=wstep
        )

        # compute number of expected windows, etc
        s_n1 = 2  # subjects in study 1
        act_n1 = 3  # activities
        tr_n1 = 2  # trials per activity
        fs1 = 100.0  # sampling rate
        ns1 = 1500  # number of samples
        s_n2 = 2
        act_n2 = 2
        tr_n2 = 3
        fs2 = 50.0
        ns2 = 1000

        N1 = int(round(ns1 * gfs / fs1))
        N2 = int(round(ns2 * gfs / fs2))

        nw1 = ((N1 - wlen_n) // wstep_n + 1) * s_n1 * act_n1 * tr_n1
        nw2 = ((N2 - wlen_n) // wstep_n + 1) * s_n2 * act_n2 * tr_n2

        n_win = nw1 + nw2
        
        assert dataset.shape == (n_win, wlen_n)
        assert 'subject_0_0' in subjects
        assert 'subject_1_0' in subjects
        assert 'subject_0_1' in subjects
        assert 'subject_1_1' in subjects


