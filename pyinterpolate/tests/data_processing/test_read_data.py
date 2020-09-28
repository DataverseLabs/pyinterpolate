import numpy as np
from pyinterpolate.data_processing.data_preparation.read_data import read_point_data
from sample_data.data import Data


def test_read_data():
    path_to_the_data = Data().gorzow_dataset
    data = read_point_data(path_to_the_data, 'txt')

    # Check if data type is ndarray
    check_ndarray = type(data) is np.ndarray

    assert check_ndarray


if __name__ == '__main__':
    test_read_data()
