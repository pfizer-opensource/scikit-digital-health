import pandas as pd
import numpy as np

from skdh.sleep.sleep_classification import compute_sleep_predictions


class TestSleepClassification:
    def test_apply_kernel(self, activity_index_data):
        # DUMMY DATA
        input_ai = pd.read_hdf(activity_index_data, key="test_ai")

        # RUN KERNEL
        output_test = compute_sleep_predictions(input_ai, rescore=False)

        # EXPECTED OUTPUT
        output_expected = (
            1 - pd.read_hdf(activity_index_data, key="expected_output").values.flatten()
        )

        assert len(output_expected) == len(output_test)
        assert np.array_equal(output_test, output_expected)

    def test_rescore(self, activity_index_data):
        # DUMMY DATA
        input_array = pd.read_hdf(activity_index_data, key="test_ai").values.flatten()

        # TEST OUTPUT (cut to sleep window)
        output_test = compute_sleep_predictions(input_array)[570:1110]

        # EXPECTED OUTPUT
        output_expected = (
            1
            - pd.read_hdf(
                activity_index_data, key="expected_output_rescored"
            ).values.flatten()
        )
        assert len(output_expected) == len(output_test)
        assert np.array_equal(output_expected, output_test)
