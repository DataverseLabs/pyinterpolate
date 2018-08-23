import numpy as np


def euclidean_metrics(points, dim):
    """
    Function calculates euclidean distance between points in n-dimensional space. Function created for
    datasets larger than 5000 rows. (It will be re-designed for parallel processing).
    :param points: numpy array with points' coordinates where each column indices new dimension and each row is
    a new coordinate set (point)
    :param dim: dimension of dataset (1 or more than 1)
    :return: distances_list - numpy array with euclidean distances between all pairs of points.
    """
    distances_list = []
    if dim == 1:
        for val in points:
            distances_list.append(np.abs(points - val))
    else:
        for row in points:
            for col in range(0, len(row)):
                single_dist_col = (points[:, col] - row[col])**2
                if col == 0:
                    multiple_distances_sum = single_dist_col
                else:
                    multiple_distances_sum = multiple_distances_sum + single_dist_col
            distances_list.append(np.sqrt(multiple_distances_sum))
    return np.array(distances_list)

def calculate_distance(points_array):
    """
    Function calculates euclidean distance between points in n-dimensional space.

    :param points_array: numpy array with points' coordinates where each column indices new dimension and each row is
    a new coordinate set (point)
    :return: distances - numpy array with euclidean distances between all pairs of points.
    
    IMPORTANT! If input array size has x rows (coordinates) then output array size is x(cols) by x(rows) 
    and each row describes distances between coordinate from row(i) with all rows. 
    The first column in row is a distance between coordinate(i) and coordinate(0), 
    the second row is a distance between coordinate(i) and coordinate(1) and so on.
    """

    try:
        number_of_cols = points_array.shape[1]
    except IndexError:
        number_of_cols = 1

    distances = euclidean_metrics(points_array, number_of_cols)

    return distances
