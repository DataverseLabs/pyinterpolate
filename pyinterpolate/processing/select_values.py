"""
Data transformation and selection.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Iterable, Dict, Union, Tuple, List

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.linalg import fractional_matrix_power

from pyinterpolate.distance.distance import calc_point_to_point_distance, calc_block_to_block_distance
from pyinterpolate.processing.preprocessing.blocks import Blocks
from pyinterpolate.processing.transform.transform import get_areal_centroids_from_agg, transform_ps_to_dict


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
    e_major_rot = [np.cos(theta), -np.sin(theta)]
    e_minor_rot = [np.sin(theta), np.cos(theta)]
    e_matrix = np.array([e_major_rot, e_minor_rot])
    return e_matrix


def _select_distances(distances_array: np.array, weighting_matrix: np.array, lag: float, step_size: float) -> np.array:
    """Function mutiplies each point from the distances array with weighting matrix to check
    if point is within ellipse.

    Parameters
    ----------
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
    WE axis (0 radian angle) of a dataset.

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
            Angle from y axis clockwise (W-E is a 0).

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


def _rotate_and_translate(points, angle, distance):
    """
    Function rotates and translates a set of points.

    Parameters
    ----------
    points : numpy array
        ``[x, y]`` coordinates.

    angle : float
        Angle of rotation in radians.

    distance : float
        The distance of translation.

    Returns
    -------
    : numpy array
        Rotated points.
    """
    points_x1 = points[:, 0] + distance * np.cos(angle)
    points_y1 = points[:, 1] + distance * np.sin(angle)

    npoints = np.column_stack((points_x1, points_y1))
    return npoints


def generate_triangles(points: np.ndarray, step_size: float, angle: float, tolerance: float) -> List:
    """Function creates triangles to select points within.

    Parameters
    ----------
    points : numpy array
        The points to find their neighbors.

    Returns
    -------
    triangles : List
        The list of triangle tuples (three coordinates per polygon and its inverted version).
        ``[triangle, inverted traingle]``

    Notes
    -----
    Each triangle width is equal to ``step_size * tolerance``, and baseline point is placed in the middle of a
    triangle's base. The height of a triangle is equal to step size. Angle points triangle to a specific direction on
    a cartesian plane.
    """

    base_width = (step_size * tolerance)
    t_height = step_size

    angle = np.radians(angle)
    rot_90 = np.pi / 2

    apex = _rotate_and_translate(points, float(angle), t_height)
    inv_apex = _rotate_and_translate(points, float(angle), -t_height)
    base_a = _rotate_and_translate(
        points, angle + rot_90, base_width
    )
    base_b = _rotate_and_translate(
        points, angle - rot_90, base_width
    )

    triangles = []
    for idx, vertex in enumerate(apex):
        triangle = (
            (base_a[idx][0], base_a[idx][1]),
            (vertex[0], vertex[1]),
            (base_b[idx][0], base_b[idx][1])
        )
        inv_triangle = (
            (base_a[idx][0], base_a[idx][1]),
            (inv_apex[idx][0], inv_apex[idx][1]),
            (base_b[idx][0], base_b[idx][1])
        )

        triangles.append([triangle, inv_triangle])

    return triangles


def select_points_within_triangle(triangle: Tuple,
                                  points: np.ndarray) -> np.ndarray:
    """
    Function selects points within a triangle.

    Parameters
    ----------
    triangle : Tuple
        ``((x1, y1), (x2, y2), (x3, y3))``

    points : numpy array
        The set of points to test.

    Returns
    -------
    : numpy array
        Boolean array of points within a triangle.
    """
    ax, ay = triangle[0]
    bx, by = triangle[1]
    cx, cy = triangle[2]

    s1 = (points[:, 0] - bx) * (ay - by) - (ax - bx) * (points[:, 1] - by)
    s2 = (points[:, 0] - cx) * (by - cy) - (bx - cx) * (points[:, 1] - cy)
    s3 = (points[:, 0] - ax) * (cy - ay) - (cx - ax) * (points[:, 1] - ay)

    s1t = s1 < 0
    s2t = s2 < 0
    s3t = s3 < 0

    stest_2a = np.logical_and.reduce((s1t, s2t, s3t))
    stest_2b = np.logical_and.reduce((~s1t, ~s2t, ~s3t))
    stest = np.logical_or(stest_2a, stest_2b)

    return stest


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
    * N - number of steps,
    * a - min_scaling_factor * value,
    * b - max_scaling_factor * value.

    Parameters
    ----------
    value : float

    min_scaling_factor : float

    max_scaling_factor : float

    number_of_steps : int

    Returns
    -------
    : numpy array
    """
    min_step = value * min_scaling_factor
    max_step = value * max_scaling_factor
    min_max_steps = np.linspace(min_step, max_step, number_of_steps)
    return min_max_steps


def get_aggregated_point_support_values(ps: Dict, indexes):
    """Function sums total point support values.

    Parameters
    ----------
    ps : Dict
         Point-support data.

    indexes : List
              List with block indexes to sum point-support values.

    Returns
    -------
    : numpy array
        Summed values of blocks' point support in the same order as given indexes List.
    """

    total_values = []
    for idx in indexes:
        _ps = ps[idx]
        tot = np.sum(_ps[:, -1])
        total_values.append(tot)

    return np.array(total_values)


def get_distances_within_unknown(point_support: np.ndarray):
    """Function calculates distances between unknown block point support points.

    Parameters
    ----------
    point_support : numpy array
                    [[x1, y1, value1], ..., [x-n, y-n, value-n]]

    Returns
    -------
    : numpy array
        [[value1, value-n, distance between points 1-n], ..., [value-n, value1, distance between points n-1]]
    """

    distances = calc_point_to_point_distance(point_support[:, :-1])
    fdistances = distances.flatten()

    values = []
    for v1 in point_support:
        for v2 in point_support:
            values.append([v1[-1], v2[-1]])

    values = np.array(values)

    values_and_distances = np.array(list(zip(values[:, 0], values[:, 1], fdistances)))

    return np.array(values_and_distances)


def get_study_max_range(input_coordinates: np.ndarray) -> float:
    """Function calculates max range of a study area.

    Parameters
    ----------
    input_coordinates : numpy array
                        [y, x] or [rows, cols]

    Returns
    -------
    study_range : float
                  It is the extent of a study area.
    """

    min_x = min(input_coordinates[:, 1])
    max_x = max(input_coordinates[:, 1])
    min_y = min(input_coordinates[:, 0])
    max_y = max(input_coordinates[:, 0])

    study_range = (max_x - min_x)**2 + (max_y - min_y)**2
    study_range = np.sqrt(study_range)
    return study_range


def prepare_pk_known_areas(point_support_dict: Dict,
                           blocks_ids: Iterable) -> Dict:
    """
    Function prepares data for semivariogram calculation between neighbors of unknown block.

    Parameters
    ----------
    point_support_dict : Dict
                         * Dict: {block id: [[point x, point y, value]]}

    blocks_ids : Iterable
                 Blocks - neighbours.

    Returns
    -------
    : Dict
        {(block a, block b): [block a value, block b value, distance between points]}
    """

    datasets = {}

    for bid_a in blocks_ids:
        ps_a = point_support_dict[bid_a]
        coordinates_a = ps_a[:, :-1]
        values_a = ps_a[:, -1]
        for bid_b in blocks_ids:
            ps_b = point_support_dict[bid_b]
            coordinates_b = ps_b[:, :-1]
            values_b = ps_b[:, -1]
            if bid_a != bid_b:
                distances = calc_point_to_point_distance(coordinates_a, coordinates_b)
            else:
                distances = np.zeros(len(values_a) * len(values_b))
            fdistances = distances.flatten()
            ldist = len(fdistances)
            a_values_arr = np.resize(values_a, ldist)
            b_values_arr = np.resize(values_b, ldist)
            out_arr = list(zip(a_values_arr, b_values_arr, fdistances))
            datasets[(bid_a, bid_b)] = np.array(out_arr)

    return datasets


def select_kriging_data(unknown_position: Iterable,
                        data_array: np.ndarray,
                        neighbors_range: float,
                        number_of_neighbors: int = 4,
                        use_all_neighbors_in_range: bool = False) -> np.ndarray:
    """
    Function prepares data for kriging - array of point position, value and distance to an unknown point.

    Parameters
    ----------
    unknown_position : Iterable
                       List, tuple or array with x, y coordinates.

    data_array : numpy array
                 Known points.

    neighbors_range : float
                      Range within neighbors are affecting the value, it should be close or the same as
                      the variogram range.

    number_of_neighbors : int, default = 4
                          Number of the n-closest neighbors used for interpolation.

    use_all_neighbors_in_range : bool, default = False
                                 True: if number of neighbors within the neighbors_range is greater than the
                                 number_of_neighbors then take all of them for modeling.

    Returns
    -------
    : numpy array
        Dataset of the length number_of_neighbors <= length. Each record is created from the position, value and
        distance to the unknown point `[[x, y, value, distance to unknown position]]`.

    """

    # Distances to unknown point
    r = np.array([unknown_position])

    known_pos = data_array[:, :-1]
    dists = calc_point_to_point_distance(r, known_pos)

    # Prepare data for kriging
    neighbors_and_dists = np.c_[data_array, dists.T]
    sorted_neighbors_and_dists = neighbors_and_dists[neighbors_and_dists[:, -1].argsort()]
    prepared_data = sorted_neighbors_and_dists[sorted_neighbors_and_dists[:, -1] <= neighbors_range, :]

    len_prep = len(prepared_data)
    is_more_than_0 = len_prep > 0

    if use_all_neighbors_in_range and is_more_than_0:
        return prepared_data

    if len_prep > number_of_neighbors:
        return sorted_neighbors_and_dists[:number_of_neighbors]
    elif not is_more_than_0:
        return sorted_neighbors_and_dists[:number_of_neighbors]
    else:
        return prepared_data


def select_poisson_kriging_data(u_block_centroid: np.ndarray,
                                u_point_support: np.ndarray,
                                k_point_support_dict: Dict,
                                nn: int,
                                max_range: float) -> Dict:
    """
    Function prepares data for the centroid-based Poisson Kriging Process.

    Parameters
    ----------
    u_block_centroid : numpy array or List
                       [index, centroid.x, centroid.y]

    u_point_support : numpy array
                      Numpy array of points within block [[x, y, point support value]]

    k_point_support_dict : Dict
                           * Dict: {block id: [[point x, point y, value]]}

    nn : int
         Maximum number of neighbours that potentially affect block.

    max_range : float
                The maximum range of influence (it should be set to semivariogram range).

    Returns
    -------
    datasets : Dict
               {known block id: [(unknown x, unknown y), [unknown val, known val, distance between points]]}
    """

    datasets = {}
    u_index = u_block_centroid[0]

    # Get closest areas
    k_idxs = list(k_point_support_dict.keys())
    distances_between_known_and_unknown = _calculate_weighted_distances(k_point_support_dict,
                                                                        u_index,
                                                                        u_point_support)
    kdata = []
    kindex = []
    for kidx in k_idxs:
        for rec in distances_between_known_and_unknown:
            if kidx in rec:
                val = rec[kidx][1]
                kdata.append(val)
                kindex.append(kidx)
                break

    kdata = np.array(kdata)
    kindex = np.array(kindex)

    sort_indices = kdata.argsort()
    sorted_kdata = kdata[sort_indices]
    sorted_kindex = kindex[sort_indices]

    max_search_pos = np.argmax(sorted_kdata > max_range)
    idxs = sorted_kindex[:max_search_pos]

    if len(idxs) < nn:
        idxs = sorted_kindex[:nn]

    if len(idxs) <= 1:
        idxs = sorted_kindex[:2]

    for idx in idxs:
        point_s = k_point_support_dict[idx]
        distances = calc_point_to_point_distance(u_point_support[:, :-1],
                                                 point_s[:, :-1])
        fdistances = distances.flatten()
        ldist = len(fdistances)
        u_coordinates_arr = [(uc[0], uc[1]) for uc in u_point_support[:, :-1]]
        u_values_arr = np.resize(u_point_support[:, -1], ldist)
        k_values_arr = np.resize(point_s[:, -1], ldist)
        u_coordinates_arr = u_coordinates_arr * int(ldist / len(u_coordinates_arr))
        out_arr = list(zip(u_values_arr, k_values_arr, fdistances))
        datasets[idx] = [u_coordinates_arr, np.array(out_arr)]

    return datasets


def select_centroid_poisson_kriging_data(u_block_centroid: np.ndarray,
                                         u_point_support: np.ndarray,
                                         k_blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                                         k_point_support_dict: Dict,
                                         nn: int,
                                         max_range: float,
                                         weighted: bool) -> np.ndarray:
    """
    Function prepares data for the centroid-based Poisson Kriging Process.

    Parameters
    ----------
    u_block_centroid : numpy array or List
                       [index, centroid.x, centroid.y]

    u_point_support : numpy array
                      Numpy array of points within block [[x, y, point support value]]

    k_blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
               Blocks with aggregated data.
               * Blocks: Blocks() class object.
               * GeoDataFrame and DataFrame must have columns: centroid.x, centroid.y, ds, index.
                 Geometry column with polygons is not used and optional.
               * numpy array: [[block index, centroid x, centroid y, value]].

    k_point_support_dict : Dict
                          * Dict: {block id: [[point x, point y, value]]}

    nn : int
         The minimum number of neighbours that potentially affect block.

    max_range : float
                The maximum range of influence (it should be set to semivariogram range).

    weighted : bool
               Are distances between blocks weighted by point support?

    Returns
    -------
    dataset : numpy array
        ``[[cx, cy, value, distance to unknown, aggregated point support sum]]``
    """

    if not isinstance(k_point_support_dict, Dict):
        k_point_support_dict = transform_ps_to_dict(k_point_support_dict)

    # Get distances from all centroids to the unknown block centroid
    k_centroids = get_areal_centroids_from_agg(k_blocks)

    u_index, u_coordinates = _transform_and_test_u_block_centroid(u_block_centroid)

    if weighted:
        # Calc weighted distance from point support
        dists = _calculate_weighted_distances(k_point_support_dict, u_index, u_point_support)
    else:
        # Calc from centroids
        dists = calc_point_to_point_distance(k_centroids[:, :-1], [u_coordinates])

    # Create Kriging Data
    kriging_indexes, kriging_data = _parse_pk_input(k_centroids, dists)

    # Sort by distance
    _s_idxs = kriging_data[:, 3].argsort()  # 3rd col == distance
    sorted_kdata = kriging_data[_s_idxs]
    sorted_indexes = kriging_indexes[_s_idxs]

    max_search_pos = np.argmax(sorted_kdata[:, 3] > max_range)

    kriging_input = sorted_kdata[:max_search_pos]

    if len(kriging_input) > nn:
        kriging_input = sorted_kdata[:nn]
        kriging_indexes = sorted_indexes[:nn]

    if len(kriging_input) <= 1:
        kriging_input = sorted_kdata[:2]
        kriging_indexes = sorted_indexes[:2]

    # get total points' value in each id from prepared datasets and append it to the array
    for idx, rec in enumerate(kriging_input):
        block_id = kriging_indexes[idx]
        points_within_block = k_point_support_dict[block_id]
        ps_total = np.sum(points_within_block[:, -1])
        kriging_input[idx][-1] = ps_total

    return kriging_input


def _calculate_weighted_distances(k_point_support_dict, u_index, u_point_support):
    dists = []

    if isinstance(u_index, np.ndarray):
        u_index = u_index[0]

    for kidx, point_array in k_point_support_dict.items():
        blocks = {
            kidx: point_array,
            u_index: u_point_support
        }
        distance = calc_block_to_block_distance(blocks)
        dists.append(distance)
    return dists


def _transform_and_test_u_block_centroid(u_block_centroid):
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
    return u_index, u_coordinates


def _parse_pk_input(centroids_and_values, distances):
    """
    Function parses given arrays into PK input.

    Parameters
    ----------
    centroids_and_values : Collection

    distances : Collection

    Returns
    -------
    : numpy array, numpy array
        indexes, [[cx, cy, value, distance to unknown centroid, 0]]
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
            centroids_and_values[:, 0],
            centroids_and_values[:, 1],
            centroids_and_values[:, 2],
            dists,
            nones
        )
    )

    data = np.array(data)
    indexes = np.array(indexes)
    return indexes, data
