import numpy as np
from scipy.linalg import fractional_matrix_power


def _rotation_matrix(angle: float) -> np.array:
    """
    Function builds rotation matrix.

    :param angle: (float) in degrees.

    :returns: (numpy array) rotation matrix.
    """
    theta = np.radians(angle)
    e_major_rot = [np.cos(theta), -np.sin(theta)]
    e_minor_rot = [np.sin(theta), np.cos(theta)]
    e_matrix = np.array([e_major_rot, e_minor_rot])
    return e_matrix


def _select_distances(distances_array, weighting_matrix, lag, step_size):
    """
    Function mutiplies each point from the distances array with weighting matrix to check if point is within an ellipse.

    :param distances_array: (numpy array) coordinates,
    :param weighting_matrix: (numpy array) matrix to perform calculations,
    :param lag: (float)
    :param step_size: (float)

    :returns: (numpy array) boolean mask of valid coordinates.
    """

    mask = []
    for pt in distances_array:
        norm_distance = np.matmul(weighting_matrix, pt)
        result = np.sqrt(norm_distance.dot(norm_distance))
        upper_limit = lag + step_size
        lower_limit = lag
        if result > 0:
            if (result <= upper_limit) and (result > lower_limit):
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(False)

    arr_mask = np.array(mask)
    return arr_mask


def select_points_within_ellipse(ellipse_center: np.array,
                                 other_points: np.array,
                                 lag: float,
                                 step_size: float,
                                 theta: float,
                                 minor_axis_size: float):
    """
    Function checks which points from other points are within point range described as an ellipse with
        center in point, semi-major axis of length step_size and semi-minor axis of length
        step_size * tolerance and angle of semi-major axis calculated as angle of direction from
        NS axis (0 radian angle) of a dataset.

    INPUT:

    :param ellipse_center: (numpy array) origin point coordinates,
    :param other_points: (numpy array),
    :param lag: (float) lag number (value),
    :param step_size: (float) step size between lags,
    :param theta: (float) Angle from y axis (N-S is a 0).
    :param minor_axis_size: (float) fraction of the major axis size.

    OUTPUT:

    :return: (numpy array) boolean array of points within ellipse with center in origin point
    """

    vector_distance = other_points - ellipse_center

    # Define Whitening Matrix
    # Lambda parameter
    e_major = 1
    e_minor = minor_axis_size
    p_lambda = np.array([[e_major, 0], [0, e_minor]])
    frac_p_lambda = fractional_matrix_power(p_lambda, -0.5)

    # Rotation matrix
    rot_matrix = _rotation_matrix(theta)

    # Whitening matrix
    w_matrix = np.matmul(frac_p_lambda, rot_matrix)

    # Distances
    current_ellipse = _select_distances(vector_distance, w_matrix, lag, step_size)

    return current_ellipse


def select_values_in_range(data, lag, step_size):
    """
    Function selects set of values which are greater than lag - step size and smaller or equal than lag.

    INPUT:
        :param data: (numpy array) distances,
        :param lag: (float) lag within areas are included,
        :param step_size: (float) step between lags.

    OUTPUT:
        :return: numpy array mask with distances within specified radius.
    """

    # Check if numpy array is given
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    greater_than = lag - step_size
    less_equal_than = lag + step_size

    # Check conditions
    condition_matrix = np.logical_and(
            np.greater(data, greater_than),
            np.less_equal(data, less_equal_than))

    # Find positions
    position_matrix = np.where(condition_matrix)
    return position_matrix
