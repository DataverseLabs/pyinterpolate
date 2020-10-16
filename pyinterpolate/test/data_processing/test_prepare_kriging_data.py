import numpy as np
from pyinterpolate.data_processing.data_transformation.prepare_kriging_data import prepare_kriging_data


def test_prepare_kriging_data():
    dataset_length = 3
    unknown_pos = (10, 10)
    pos = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [13, 10, 9]]
    pos = np.array(pos)
    d = prepare_kriging_data(unknown_pos, pos, dataset_length)
    vals = np.array([[13, 10, 9, 3], [4, 4, 4, 8], [3, 3, 3, 9]])
    d = d.astype(np.int)

    test_output = (d == vals).all()

    test_length = len(d) == dataset_length

    assert (test_length and test_output)


if __name__ == '__main__':
    test_prepare_kriging_data()
