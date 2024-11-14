import numpy as np
from scipy.linalg import fractional_matrix_power


def define_whitening_matrix(theta: float, minor_axis_size: float):
    """
    Function defines whitening matrix.

    Parameters
    ----------
    theta : float
        Angle from y axis counterclockwise (W-E is a 0).

    minor_axis_size : float
        Fraction of the major axis size.

    Returns
    -------
    w_matrix : numpy array
    """
    # Lambda parameter
    e_major = 1
    e_minor = minor_axis_size
    p_lambda = np.array([[e_major, 0],
                         [0, e_minor]])
    frac_p_lambda = fractional_matrix_power(p_lambda, -0.5)

    # Rotation matrix
    rot_matrix = _rotation_matrix(theta)

    # Whitening matrix
    w_matrix = np.matmul(frac_p_lambda, rot_matrix)
    return w_matrix


def select_points_within_ellipse(vector_distances: np.array,
                                 lag: float,
                                 step_size: float,
                                 w_matrix: np.ndarray) -> np.array:
    """Function checks which points from other points are within the point
    range described as an ellipse with center in the point, the semi-major axis
    of the ``step_size`` length, and the semi-minor axis of length
    ``step_size * tolerance``. The direction angle of semi-major starts from
    W-E axis (0 radian direction) and goes counterclockwise.

    Parameters
    ----------
    vector_distances : numpy array
        Distances between all points.

    lag : float

    step_size : float
        Step size between lags.

    w_matrix : numpy array
        Matrix used for masking values in ellipse.

    Returns
    -------
    : numpy array
        Boolean array of points within ellipse with a center in origin point.
    """
    # Distances
    current_ellipse = _select_ellipse_distances(vector_distances,
                                                w_matrix,
                                                lag,
                                                step_size)

    return current_ellipse


def _rotation_matrix(angle: float) -> np.array:
    """Function builds rotation matrix.

    Parameters
    ----------
    angle : float
            Angle in degrees.

    Returns
    -------
    : numpy array
        The rotation matrix.
    """
    theta = np.radians(angle)
    e_major_rot = [np.cos(theta), np.sin(theta)]
    e_minor_rot = [-np.sin(theta), np.cos(theta)]
    e_matrix = np.array([e_major_rot, e_minor_rot])
    return e_matrix


def _select_ellipse_distances(distances_array: np.array,
                              weighting_matrix: np.array,
                              lag: float,
                              step_size: float) -> np.array:
    """Function mutiplies each point from the distances array with
    a weighting matrix to check if point is within the ellipse.

    Parameters
    ----------
    distances_array : numpy array
        Array with distances between points.

    weighting_matrix : numpy array
        The matrix of custom_weights for each distance.

    lag : float

    step_size : float

    Returns
    -------
    : numpy array
        Boolean mask of valid coordinate indexes.
    """
    norm_distances = np.matmul(weighting_matrix, distances_array.T).T
    norm_results = norm_distances.dot(norm_distances.T).diagonal()
    norm_results = np.sqrt(norm_results)
    upper_limit = lag
    lower_limit = lag - step_size

    mask_a = norm_results <= upper_limit
    mask_b = norm_results > lower_limit
    mask_c = norm_results != 0

    mask_out = mask_a & mask_b & mask_c

    return mask_out
