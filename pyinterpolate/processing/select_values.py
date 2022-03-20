import numpy as np
from scipy.linalg import fractional_matrix_power


def _rotation_matrix(angle: float) -> np.array:
    """Function builds rotation matrix.

    Parameters
    ----------
    angle : float
            Angle in degrees.

    Returns
    -------
    numpy array
        The rotation matrix.
    """
    theta = np.radians(angle)
    e_major_rot = [np.cos(theta), -np.sin(theta)]
    e_minor_rot = [np.sin(theta), np.cos(theta)]
    e_matrix = np.array([e_major_rot, e_minor_rot])
    return e_matrix


def _select_distances(distances_array: np.array, weighting_matrix: np.array, lag: float, step_size: float) -> np.array:
    """Function mutiplies each point from the distances array with weighting matrix to check
       if point is within ellipse.

    Parameters
    distances_array : numpy array
                      Array with distances between points.

    weighting_matrix : numpy array
                       The matrix of weights for each distance.

    lag : float

    step_size : float

    Returns
    -------
    : numpy array
        Boolean mask of valid coordinate indexes.
    """

    mask = []
    for pt in distances_array:
        norm_distance = np.matmul(weighting_matrix, pt)
        result = np.sqrt(norm_distance.dot(norm_distance))
        upper_limit = lag
        lower_limit = lag - step_size
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
                                 minor_axis_size: float) -> np.array:
    """Function checks which points from other points are within point range described as an ellipse with
        center in point, semi-major axis of length step_size and semi-minor axis of length
        step_size * tolerance and angle of semi-major axis calculated as angle of direction from
        NS axis (0 radian angle) of a dataset.

    Parameters
    ----------
    ellipse_center : numpy array
                     Origin point coordinates.

    other_points : numpy array
                   Array with points for which distance is calculated.

    lag : float

    step_size : float
                Step size between lags.

    theta : float
            Angle from y axis clockwise (N-S is a 0).

    minor_axis_size : float
                      Fraction of the major axis size.

    Returns
    -------
    : numpy array
        Boolean array of points within ellipse with a center in origin point.
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
    """Function selects set of values which are greater than lag - step size and smaller or equal than lag.

    Parameters
    ----------
    data : numpy array
           Distances between points.

    lag : float

    step_size : float
                Distance between lags.

    Returns
    -------
    : numpy array
        Mask with distances within a specified radius.
    """

    # Check if numpy array is given
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    greater_than = lag - step_size
    less_equal_than = lag

    # Check conditions
    condition_matrix = np.logical_and(
            np.greater(data, greater_than),
            np.less_equal(data, less_equal_than))

    # Find positions
    position_matrix = np.where(condition_matrix)
    return position_matrix


def create_min_max_array(value: float,
                         min_scaling_factor: float,
                         max_scaling_factor: float,
                         number_of_steps: int) -> np.array:
    """Function prepares a numpy array of N equidistant values between (a:b), where:
        N - number of steps,
        a - min_scaling_factor * value
        b - max_scaling_factor * value

    Parameters
    ----------
    value : float

    min_scaling_factor : float

    max_scaling_factor : float

    number_of_steps : int

    Returns
    -------
    numpy array

    """
    min_step = value * min_scaling_factor
    max_step = value * max_scaling_factor
    min_max_steps = np.linspace(min_step, max_step, number_of_steps)
    return min_max_steps
