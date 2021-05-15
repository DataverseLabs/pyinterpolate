import numpy as np
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance


def calculate_semivariance_within_blocks(points_within_area, semivariance_model):
    """
    Function calculates semivariances of points inside all given areal blocks.

    gamma(v, v) = 1/(P * P) * SUM(s->P) SUM(s'->P) gamma(u_s, u_s')
    where:
    gamma(v, v) - average within block semivariance,
    P - number of points used to discretize block v,
    u_s - point u within block v,
    gamma(u_s, u_s') - semivariance between point u_s and u_s' inside block v.

    :param points_within_area: (numpy array / list of lists) [area_id, array of points within area and their values],
    :param semivariance_model: (TheoreticalSemivariogram) Theoretical Semivariogram object,
    :return within_areas_semivariance: (numpy array) [area_id, semivariance]
    """

    within_areas_semivariance = []

    for block in points_within_area:

        # Get area id
        area_id = block[0]

        # Calculate inblock semivariance for a given id
        number_of_points_within_block = len(block[1])  # P
        p = number_of_points_within_block * number_of_points_within_block

        distances_between_points = calc_point_to_point_distance(block[1][:, :-1])

        semivariances = semivariance_model.predict(distances_between_points)

        avg_semivariance = np.sum(semivariances) / p
        within_areas_semivariance.append([area_id, avg_semivariance])

    return within_areas_semivariance
