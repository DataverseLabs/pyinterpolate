import numpy as np


def check_points_within_ellipse(origin_point: np.array, other_points: np.array, step_size: float,
                                last_step_size: float, angle: float, tolerance: float):
    """
    Function checks which points from other points are within point range described as an ellipse with
        center in point, semi-major axis of length step_size and semi-minor axis of length
        step_size * tolerance and angle of semi-major axis calculated as angle of direction from
        NS axis (0 radian angle) of a dataset.

    INPUT:

    :param origin_point: (numpy array) single point coordinates,
    :param other_points: (numpy array),
    :param step_size: (float) current distance between lags within each points are included in the calculations,
    :param last_step_size: (float) last distance between lags within each points are included in the calculations,
    :param angle: (float) direction of semivariogram, in radians. Angle is rotated by PI/2 rad.
    :param tolerance: (float) value in range 0-1 normalized to [0 : 0.5] to select tolerance of semivariogram. If
        tolerance is 0 then points must be placed at a single line with beginning in the origin of coordinate
        system and angle given by y axis and direction parameter. If tolerance is greater than 0 then semivariance
        is estimated from elliptical area with major axis with the same direction as the line for 0 tolerance
        and minor axis of a size:

        (tolerance * step_size)

        and major axis (pointed in NS direction):

        ((1 - tolerance) * step_size)

        and baseline point at a center of ellipse. Tolerance == 1 creates omnidirectional semivariogram.

    ROTATED ELLIPSE EQUATION:

        part_a = (cos(A) * (x - h) + sin(A) * (y - k))**2
        part_b = (sin(A) * (x - h) + cos(A) * (y - k))**2

        and if:

            part_a / r_x**2 + part_b / r_y**2 <= 1

        then point is inside ellipse.

    OUTPUT:

    :return: (numpy array) boolean array of points within ellipse with center in origin point
    """

    rx_base = (step_size * tolerance) ** 2
    ry_base = (step_size * (1 - tolerance)) ** 2

    rx_prev = (last_step_size * tolerance) ** 2
    ry_prev = (last_step_size * (1 - tolerance)) ** 2

    bool_mask = []

    for point in other_points:

        try:
            is_origin = (point == origin_point).all()
        except AttributeError:
            is_origin = point == origin_point

        if is_origin:
            bool_mask.append(False)
        else:
            if ry_base == 0:
                part_a_base = 0
                part_a_previous = 0
            else:
                part_a_x = (point[1] - origin_point[1]) * np.cos(angle)
                part_a_y = (point[0] - origin_point[0]) * np.sin(angle)

                # Points within base
                part_a_base = (part_a_x + part_a_y) ** 2 / ry_base

                # Points within previous ellipse
                part_a_previous = (part_a_x + part_a_y) ** 2 / ry_prev

            if rx_base == 0:
                part_b_base = 0
                part_b_previous = 0
            else:
                part_b_x = (point[1] - origin_point[1]) * np.sin(angle)
                part_b_y = (point[0] - origin_point[0]) * np.cos(angle)

                # Points within base
                part_b_base = (part_b_x + part_b_y) ** 2 / rx_base

                # Points within previous ellipse
                part_b_previous = (part_b_x + part_b_y) ** 2 / rx_prev

            # Points within base
            test_value_base = part_a_base + part_b_base

            # Points within previous ellipse
            test_value_prev = part_a_previous + part_b_previous

            if last_step_size == 0:
                # This is the first step of analysis
                if test_value_base <= 1:
                    bool_mask.append(True)
                else:
                    bool_mask.append(False)
            else:
                # Second and next steps of analysis

                # If point is within big ellipse and it is not in the previous ellipse
                if test_value_base <= 1 < test_value_prev:
                    bool_mask.append(True)
                else:
                    bool_mask.append(False)

    return np.array(bool_mask)


def select_values_in_range(data, lag, step_size):
    """
    Function selects set of values which are greater than (lag - step size / 2) and smaller or equal than
        (lag + step size / 2).

    INPUT:

    :param data: (numpy array) distances,
    :param lag: (float) lag within areas are included,
    :param step_size: (float) step between lags.

    OUTPUT:

    :return: numpy array mask with distances within specified radius.
    """

    step_size = step_size / 2

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
