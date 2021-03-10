import numpy as np


def read_point_data(path, data_type):
    """
    Function reads data from a text file and converts it into numpy array.

    INPUT:

    :param path: (str) path to the file,
    :param data_type: (str) data type, available types:
        - 'txt' for txt files.

    OUTPUT:

    :return data_arr: (numpy array) of coordinates and their values."""
    if data_type == 'txt':
        data_arr = np.loadtxt(path, delimiter=',')
        return data_arr
    
    raise ValueError('Data type not supported')
