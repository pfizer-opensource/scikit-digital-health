import numpy as np

from skimu.sleep.endpoints import sleep_average_hazard


class TestSleepAverageHazard:
    def test(self):
        x = np.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             0]
        )

        sh = sleep_average_hazard(x)

        assert np.isclose(sh, 0.8333333)
