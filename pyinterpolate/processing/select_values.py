from typing import Iterable, Dict

import numpy as np
from scipy.linalg import fractional_matrix_power

from pyinterpolate.distance.distance import calc_point_to_point_distance, calc_block_to_block_distance


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


def get_study_max_range(input_coordinates: np.ndarray) -> float:
    """Function calculates max range of a study area.

    Parameters
    ----------
    input_coordinates : numpy array

    Returns
    -------
    study_range : float
    """

    min_x = min(input_coordinates[:, 1])
    max_x = max(input_coordinates[:, 1])
    min_y = min(input_coordinates[:, 0])
    max_y = max(input_coordinates[:, 0])

    study_range = (max_x - min_x)**2 + (max_y - min_y)**2
    study_range = np.sqrt(study_range)
    return study_range


def select_kriging_data(unknown_position: Iterable,
                        data_array: np.ndarray,
                        neighbors_range: float,
                        min_number_of_neighbors: int = 4,
                        max_number_of_neighbors: int = -1) -> np.ndarray:
    """
    Function prepares data for kriging - array of point position, value and distance to an unknown point.

    Parameters
    ----------
    unknown_position : Iterable
                       List, tuple or array with x, y coordinates.

    data_array : np.ndarray
                 Known points.

    neighbors_range : float
                      Range within neighbors are affecting the value, it should be close or the same as
                      the variogram range.

    min_number_of_neighbors : int, default = 4
                              Number of the n-closest neighbors used for interpolation. If within the
                              neighbors_range is less neighbors than min_number_of_neighbors, then additional points are
                              selected from outside the neighbors_range based on their position.

    max_number_of_neighbors : int, default = -1
                              Maximum number of neighbors within neighbors_range. You should leave default value
                              -1 if you want to include all available points. If your range of analysis catches multiple
                              points, you may consider to set this parameter to some large integer value to speed-up
                              the computations.

    Returns
    -------
    : np.ndarray
      Dataset of the length min_number_of_neighbors <= length <= max_number_of_neighbors. Each record is created from
      the position, value and distance to the unknown point `[[x, y, value, distance to unknown position]]`.

    """

    # Distances to unknown point
    r = np.array([unknown_position])

    known_pos = data_array[:, :-1]
    dists = calc_point_to_point_distance(r, known_pos)

    # Prepare data for kriging
    neighbors_and_dists = np.c_[data_array, dists.T]
    prepared_data = neighbors_and_dists[neighbors_and_dists[:, -1] <= neighbors_range, :]

    len_prep = len(prepared_data)

    if max_number_of_neighbors == -1:
        # Get all neighbors
        return prepared_data

    if len_prep < min_number_of_neighbors:
        # Get minimal number of neighbors
        sorted_neighbors_and_dists = neighbors_and_dists[neighbors_and_dists[:, -1].argsort()]
        prepared_data = sorted_neighbors_and_dists[:min_number_of_neighbors]
    else:
        # Get max number of neighbors
        sorted_neighbors_and_dists = neighbors_and_dists[neighbors_and_dists[:, -1].argsort()]
        prepared_data = sorted_neighbors_and_dists[:max_number_of_neighbors]

    return prepared_data


def select_poisson_kriging_data(u_block_centroid: np.ndarray,
                                u_point_support: np.ndarray,
                                k_blocks: Dict,
                                k_point_support: Dict,
                                nn: int,
                                max_radius: float,
                                weighted: bool) -> np.ndarray:
    """
    Function prepares data for the Poisson Kriging Process.

    Parameters
    ----------
    u_block_centroid : numpy array or List
                       [index, centroid.x, centroid.y]

    u_point_support : numpy array
                      Numpy array of points within block [[x, y, point support value]]

    k_blocks : Dict
               Dictionary retrieved from the Blocks, it's structure is defined as:
               polyset = {
                      'geometry': {
                          'block index': geometry
                      }
                      'data': [[index centroid.x, centroid.y value]],
                      'info': {
                          'index_name': the name of the index column,
                          'geometry_name': the name of the geometry column,
                          'value_name': the name of the value column,
                          'crs': CRS of a dataset
                      }
                  }

    k_point_support : Dict
                      Point support data as a Dict:

                        point_support = {
                            'area_id': [numpy array with points]
                        }

    nn : int
         The minimum number of neighbours that potentially affect block.

    max_radius : float
                 The maximum radius of search for the closest neighbors.

    weighted : bool
               Are distances between blocks weighted by point support?

    Returns
    -------
    dataset : numpy array
              [block id, cx, cy, value, distance to unknown, aggregated point support sum]
    """

    # Get distances from all centroids to the unknown block centroid
    k_centroids = np.array([x[1:] for x in k_blocks['data']])

    if not isinstance(u_block_centroid, np.ndarray):
        u_block_centroid = np.array(u_block_centroid)

    if len(u_block_centroid) != 3:
        u_block_centroid = u_block_centroid.flatten()
        if len(u_block_centroid) != 3:
            raise AttributeError(
                f'Parameter u_block_centroid should have three records: index, coordinate x, coordinate y. '
                f'But provided array has {len(u_block_centroid)} record(s)!')

    u_coordinates = u_block_centroid[1:]
    u_index = u_block_centroid[0]

    dists = []

    if weighted:
        # Calc from point support
        for kidx, point_array in k_point_support['data'].items():
            blocks = {
                kidx: point_array,
                u_index: u_point_support
            }
            distance = calc_block_to_block_distance(blocks)
            dists.append(distance)
    else:
        # Calc from centroids
        dists = calc_point_to_point_distance(k_centroids[:, :-1], [u_coordinates])

    # Create Kriging Data
    kriging_data = _parse_pk_input(k_centroids, dists)

    # Sort by distance
    kriging_data = kriging_data[kriging_data[:, 4].argsort()]  # 4th idx == distance

    # Get distances in max search radius
    max_search_pos = np.argmax(kriging_data[:, 4] > max_radius)
    kriging_input = kriging_data[:max_search_pos]

    # check number of observations
    if len(kriging_input) < nn:
        kriging_input = kriging_data[:nn]

    # get total points' value in each id from prepared datasets and append it to the array

    for idx, rec in enumerate(kriging_input):
        block_id = rec[0]
        try:
            points_within_block = k_point_support['data'][block_id]
        except KeyError as kex:
            if isinstance(k_point_support, Dict):
                points_within_block = k_point_support[block_id]
            else:
                raise kex
        ps_total = np.sum(points_within_block[:, -1])
        kriging_input[idx][-1] = ps_total

    return kriging_input


def _parse_pk_input(centroids_and_values, distances):
    """
    Function parses given arrays into PK input.

    Parameters
    ----------
    centroids_and_values : Collection

    distances : Collection

    Returns
    -------
    : numpy array
        [[id, cx, cy, value, distance to unknown centroid, 0]]
    """
    indexes = []
    dists = []

    if isinstance(distances[0], dict):
        dists = []
        for rec in distances:
            k0 = list(rec.keys())[0]
            dists.append(
                rec[k0][1]
            )
            indexes.append(k0)
    elif isinstance(distances, np.ndarray):
        indexes = [x[0] for x in centroids_and_values]
        dists = [x[0] for x in distances]

    nones = [0 for _ in indexes]

    data = list(
        zip(
            indexes,
            centroids_and_values[:, 0],
            centroids_and_values[:, 1],
            centroids_and_values[:, 2],
            dists,
            nones
        )
    )

    data = np.array(data)
    return data
