import numpy as np
from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance


def calculate_inblock_semivariance(points_within_area, semivariance_model):
    """
    Function calculates semivariances of points inside all given areal blocks.
    :param points_within_area: (numpy array / list of lists) [area_id, array of points within area and their values],
    :param semivariance_model: (TheoreticalSemivariogram) Theoretical Semivariogram object,
    :return areas_inblock_semivariance: (numpy array) [area_id, inblock_semivariance]
    """

    areas_inblock_semivariance = []
    for block in points_within_area:

        # Get area id
        area_id = block[0]

        # Calculate inblock semivariance for a given id
        number_of_points_within_block = len(block[1])
        squared_no_points = number_of_points_within_block * number_of_points_within_block
        semivars = []
        for point in block[1][:, :-1]:
            avg_sem = []
            for point2 in block[1][:, :-1]:
                dist = calc_point_to_point_distance([point], [point2])
                smv = semivariance_model.predict(dist)
                avg_sem.append(smv)
            semivars.append(np.mean(avg_sem))
        avg_inblock_semivariance = np.sum(semivars) / squared_no_points

        areas_inblock_semivariance.append([area_id, avg_inblock_semivariance])
    return areas_inblock_semivariance
