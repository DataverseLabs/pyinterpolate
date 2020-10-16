import numpy as np
from pyinterpolate.data_processing.data_preparation.select_values_in_range import select_values_in_range


def test_select_values_in_range():
    dataset = [[1, 2, 3],
               [4, 5, 6],
               [1, 9, 9]]
    output = (np.array([1, 1, 1]), np.array([0, 1, 2]))
    x = select_values_in_range(dataset, 5, 2)

    assert (x[0] == output[0]).all()
    assert (x[1] == output[1]).all()


if __name__ == '__main__':
    test_select_values_in_range()
