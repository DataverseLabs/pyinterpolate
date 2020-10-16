import numpy as np
from pyinterpolate.data_processing.data_preparation.read_data import read_point_data


def test_read_data():
    path_to_the_data = 'sample_data/poland_dem_gorzow_wielkopolski'
    data = read_point_data(path_to_the_data, 'txt')

    # Check if data type is ndarray
    check_ndarray = isinstance(data, np.ndarray)

    assert check_ndarray

    # Check dimensions

    assert data.shape[1] == 3


if __name__ == '__main__':
    test_read_data()
