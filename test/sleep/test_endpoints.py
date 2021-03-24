import numpy as np

from skimu.sleep.utility import rle
from skimu.sleep.endpoints import SleepAverageHazard


class TestSleepAverageHazard:
    def test(self):
        x = np.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             0]
        )
        l, s, v = rle(x)

        sh = SleepAverageHazard().predict(lengths=l, starts=s, values=v)

        assert np.isclose(sh, 0.8333333)
