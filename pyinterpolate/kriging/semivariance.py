import numpy as np
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance


def calculate_semivariance(points_array, lags, step_size):
    """
    Function calculates semivariance of points in n-dimensional space.
    :param points_array: numpy array of points and values (especially DEM) where
    points_array[0] = array([point_x, point_y, ..., point_n, value])
    :param lags: array of lags between points
    :param step_size: distance which should be included in the gamma parameter which enhances range of interest
    :return: semivariance: numpy array of pair of lag and semivariance values where
    semivariance[0] = array of lags
    semivariance[1] = array of lag's values
    semivariance[2] = array of number of points in each lag

    """
    smv = []
    semivariance = []

    # Calculate distance
    distance_array = calculate_distance(points_array[:, 0:-1])

    # Calculate semivariances
    for h in lags:
        gammas = []
        distances_in_range = np.where(
            np.logical_and(
                np.greater_equal(distance_array, h - step_size), np.less_equal(distance_array, h + step_size)))
        for i in range(0, len(distances_in_range[0])):
            row_x = distances_in_range[0][i]
            row_x_h = distances_in_range[1][i]
            gp1 = points_array[row_x][-1]
            gp2 = points_array[row_x_h][-1]
            g = (gp1 - gp2) ** 2
            gammas.append(g)
        if len(gammas) == 0:
            gamma = 0
        else:
            gamma = np.sum(gammas) / (2 * len(gammas))
        smv.append([gamma, len(gammas)])

    # Selecting semivariances
    for i in range(len(lags)):
        if smv[i][0] > 0:
            semivariance.append([lags[i], smv[i][0], smv[i][1]])

    semivariance = np.vstack(semivariance)

    return semivariance.T


def calculate_inblock_semivariance(blocks_dict):
    """
    Function calculates semivariance of points inside a block (area).
    :param blocks_dict: block dictionary in the minimal form: {block id: 
                                                                {'coordinates': [[x0, y0, val0],
                                                                                 [x1, y1, val1],
                                                                                 [x.., y.., val..]]
                                                                }
                                                              }
    :return: updated block dictionary with new key 'semivariance' and mean variance per area
    """
    semivariance = []
    
    blocks = list(blocks_dict.keys())

    for block in blocks:
        
        points_array = np.asarray(blocks_dict[block]['coordinates'])
        
        # Calculate semivariance
        number_of_points = len(points_array)
        p_squared = number_of_points**2
        
        for point1 in points_array:
            variances = []
            for point2 in points_array:
                v = point1[-1] - point2[-1]
                v = (v**2)
                variances.append(v)
            variance = np.sum(variances) / (2 * len(variances))
            semivariance.append(variance)
        
        semivar = np.sum(semivariance) / p_squared
        
        blocks_dict[block]['semivariance'] = semivar

    return blocks_dict
