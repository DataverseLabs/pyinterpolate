import numpy as np


def get_total_value_of_area(areal_points):
    total = np.sum(areal_points[1][:, 2])
    return total

def set_areal_weights(areal_data, areal_points):
    """
    Function prepares array for weighted semivariance calculation.

    INPUT:

    :param areal_data: (numpy array) of areas in the form:
        [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value],
    :param areal_points: (numpy array) of points within areas in the form:
        [area_id, [point_position_x, point_position_y, value]].

    OUTPUT:

    :return: (numpy array) of weighted points.
    """

    weighted_semivariance_input = []
    for rec in areal_data:
        rec_id = rec[0]

        # Calculate total value of points within area
        total = 0
        for points_rec in areal_points:
            if points_rec[0] == rec_id:
                total = get_total_value_of_area(points_rec)
                break
            else:
                pass

        output_record = [rec[2], rec[3], rec[4], total]
        weighted_semivariance_input.append(output_record)

    weighted_semivariance_input = np.array(weighted_semivariance_input)
    return weighted_semivariance_input
