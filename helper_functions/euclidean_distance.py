import numpy as np


def calculate_distance(points_array):
    """
    Function for calcultaing euclidean distance between points in n-dimensional space.

    :param points_array: numpy array with points' coordinates where each column indices new dimension and each row is
    a new coordinate set (point)
    :return: distances - numpy array with euclidean distances between all pairs of points.
    
    IMPORTANT! If input array size has x rows (coordinates) then output array size is x(cols) by x(rows) 
    and each row describes distances between coordinate from row(i) with all rows. 
    The first column in row is a distance between coordinate(i) and coordinate(0), 
    the second row is a distance between coordinate(i) and coordinate(1) and so on.
    """
    points_dictionary = {}
    maximum_length = 5000
    number_of_rows = points_array.shape[0]
    distances = []

    if len(points_array.shape) == 1:
        arr = points_array.ravel()
        if len(arr) > maximum_length:
            raise ValueError('Please provide array with less than 5000 elements')
        else:
            points_dictionary[1] = arr
    else:
        for i in range(points_array.shape[1]):
            dimension = i + 1
            arr = (points_array[:, i]).ravel()
            if len(arr) > maximum_length:
                raise ValueError('Please provide array with less than 5000 elements')
            points_dictionary[dimension] = arr

    if len(points_dictionary) == 1:
        distances = np.subtract.outer(points_dictionary[1], points_dictionary[1]).ravel()
        distances = np.abs(distances)
    elif len(points_dictionary) > 1:
        for key in points_dictionary:
            dist = np.subtract.outer(points_dictionary[key], points_dictionary[key]).ravel()
            dist = dist ** 2
            if key == 1:
                distances = dist
            else:
                distances += dist
        distances = np.sqrt(distances)
    else:
        raise ValueError('Something is wrong. Did you pass an empty array?')
    return distances.reshape(number_of_rows, number_of_rows)
