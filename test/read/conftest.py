from pytest import fixture
from numpy import zeros, array


@fixture
def windowing_data():
    def get_data(case):
        n_samp = 300
        ind = zeros(50) - 600

        if case == '24hr':
            ind[[10, 17, 31, 43]] = [6, 15, 29, 44]

            starts = array([0, 6, 15, 29, 44])
            stops = array([6, 15, 29, 44, n_samp])
        elif case == 'full first, full last':
            ind[[10, 17, 31, 43]] = [6, -15, 29, -44]

            starts = array([6, 29])
            stops = array([15, 44])
        elif case == 'full first, partial last':
            ind[[10, 17, 31, 43, 47]] = [6, -15, 29, -44, 60]

            starts = array([6, 29, 60])
            stops = array([15, 44, n_samp])
        elif case == 'partial first, full last':
            ind[[10, 17, 31, 43, 47]] = [-6, 15, -29, 44, -80]

            starts = array([0, 15, 44])
            stops = array([6, 29, 80])
        elif case == 'partial first, partial last':
            ind[[10, 17, 31, 43]] = [-6, 15, -29, 44]

            starts = array([0, 15, 44])
            stops = array([6, 29, n_samp])

        return (ind, n_samp), (starts, stops)

    return get_data
