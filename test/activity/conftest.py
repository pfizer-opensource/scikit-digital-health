from pytest import fixture
import numpy as np


@fixture(scope="module")
def get_sample_activity_bout_data():
    def get_data(boutmetric, boutdur):
        rr1 = np.zeros(500)
        rr1[49:90] = 1
        rr1[99:115] = 1
        rr1[199:300] = 1
        rr1[307:315] = 1
        rr1[319:360] = 1
        rr1[372:450] = 1

        ret = (rr1,)
        if boutdur == 1:
            if boutmetric == 1:
                ret += (23.75,)
            elif boutmetric == 2:
                ret += (24.333333333333,)
            elif boutmetric == 3:
                ret += (24.75,)
            elif boutmetric == 4:
                ret += (23.0833333333333,)
            elif boutmetric == 5:
                ret += (23.0833333333333,)
        elif boutdur == 2:
            if boutmetric == 1:
                ret += (23.75,)
            elif boutmetric == 2:
                ret += (24.4166666666666,)
            elif boutmetric == 3:
                ret += (25.0833333333333,)
            elif boutmetric == 4:
                ret += (22.75,)
            elif boutmetric == 5:
                ret += (22.75,)

        return ret
    return get_data
