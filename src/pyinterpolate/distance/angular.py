import concurrent.futures
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import fractional_matrix_power


def build_mask_indices(coordinates: np.ndarray, vertices: List):
    """
    Function builds masks for points within edges forming triangles.

    Parameters
    ----------
    coordinates : numpy array
        Masked points.

    vertices : List
        Triangle edges.

    Returns
    -------
    : List
        List of length ``edges`` with point-indices.
    """

    # Create boolean mask for each lag and coordinate and get True indices
    indices = []

    def _get(lag_idx):

        point_masks = []

        trs = vertices[lag_idx]
        for tr in trs:
            mask = triangle_mask(
                triangle_1=tr[0],
                triangle_2=tr[1],
                coordinates=coordinates
            )
            point_masks.append(mask)

        indices.append([lag_idx, point_masks])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(len(vertices)):
            futures.append(
                executor.submit(
                    _get, idx
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                raise e

    indices = sorted(
        indices,
        key=lambda x: x[0]
    )

    return indices


def calc_angles_between_points(vec1: np.ndarray,
                               vec2: np.ndarray,
                               flatten_output: bool = True) -> np.ndarray:
    """
    Function calculates distances between two groups of points as their cross
    product.

    Parameters
    ----------
    vec1 : numpy array
        The first set of coordinates.

    vec2 : numpy array
        The second set of coordinates.

    flatten_output : bool, default = True
        Return flattened array (vector).

    Returns
    -------
    angles : numpy array
        An array with angles between all points from ``vec1`` to all
        points from ``vec2``, where rows are angles between
        points from ``vec1`` to points from ``vec2`` (columns).
    """
    angles = []

    for point in vec1:
        row = calc_angles(vec2, origin=point)
        angles.append(row.flatten())

    angles = np.array(angles)

    if flatten_output:
        return angles.flatten()

    return angles


def calc_angles(points_b: ArrayLike,
                point_a: ArrayLike = None,
                origin: ArrayLike = None) -> np.ndarray:
    """
    Function calculates angles between points and origin or between
    vectors from origin to points and a vector from a specific point to
    origin.

    Parameters
    ----------
    points_b : numpy array
        Other point coordinates.

    point_a : Iterable
        The point coordinates, default is equal to (0, 0).

    origin : Iterable
        The origin coordinates, default is (0, 0).

    Returns
    -------
    angles : numpy array
        Angles from the ``points_b`` to origin, or angles between vectors
        ``points_b`` to origin and ``point_a`` to origin.
    """

    if origin is None:
        origin = np.array((0, 0))
    else:
        if not isinstance(origin, np.ndarray):
            origin = np.array(origin)

    if not isinstance(points_b, np.ndarray):
        points_b = np.array(points_b)

    if len(points_b.shape) == 1:
        points_b = points_b[np.newaxis, ...]

    if point_a is None:
        angles = _calc_angle_from_origin(points_b, origin)
    else:
        angles = _calc_angle_between_points(point_a, points_b, origin)

    return angles


def calculate_angular_difference(angles: np.ndarray,
                                 expected_direction: float) -> np.ndarray:
    """
    Function calculates difference between an angle given by vector crossing
    origin and location coordinates and expected direction of directional
    variogram.

    Parameters
    ----------
    angles : numpy array
        Angles between vectors from origin to points and x-axis on cartesian
        plane.

    expected_direction : float
        The variogram direction in degrees.

    Returns
    -------
    normalized_angular_diffs : numpy array
        Minimal direction from ``expected_direction`` to other angles.
    """

    # We should select angles equal to the expected direction
    # and 180 degrees from it

    expected_direction_rad = np.deg2rad(expected_direction)
    r_angles = np.deg2rad(angles)
    norm_a = r_angles - expected_direction_rad
    deg_norm_a = np.abs(np.rad2deg(norm_a % (2 * np.pi)))
    norm_b = expected_direction_rad - r_angles
    deg_norm_b = np.abs(np.rad2deg(norm_b % (2 * np.pi)))
    normalized_angular_diffs = np.minimum(deg_norm_a, deg_norm_b)

    return normalized_angular_diffs


def clean_mask_indices(indices: List):
    """
    Function cleans a lag masks from the previous lags masks in the same
    location.

    Parameters
    ----------
    indices : List
        Lag-ordered list of point-masks.

    Returns
    -------
    cleaned_indices : List
        Lag-ordered list of cleaned point-masks.
    """
    cleaned_masks = [
        indices[0][1]
    ]

    for lag_idx, lag_masks in enumerate(indices[1:]):
        masks = lag_masks[1]
        previous_masks = indices[lag_idx][1]
        new_masks = []
        for idx, mask in enumerate(masks):
            new_mask = np.logical_xor(
                mask, previous_masks[idx]
            )
            new_masks.append(new_mask)
        cleaned_masks.append(new_masks)

    return cleaned_masks


def define_whitening_matrix(theta: float,
                            minor_axis_size: float) -> np.ndarray:
    """
    Function defines whitening matrix.

    Parameters
    ----------
    theta : float
        Angle from y-axis counterclockwise (W-E is a 0).

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


def filter_triangles_mask(old_mask, new_mask) -> np.array:
    """
    Function checks masks with larger radius to filter values that are set to
    be a mask in the lag-1 from values that are set to be a mask in the lag+0.

    Parameters
    ----------
    old_mask : numpy array
        Boolean mask for the previous lag.

    new_mask : numpy array
        Boolean mask for the current lag.

    Returns
    -------
    : numpy array
        New boolean mask (XOR ops).
    """
    mask = []
    for idx, val in enumerate(new_mask):
        if old_mask[idx]:
            mask.append(False)
        else:
            if val:
                mask.append(True)
            else:
                mask.append(False)
    return np.array(mask)


def generate_triangles(points: np.ndarray,
                       step_size: float,
                       angle: float,
                       tolerance: float) -> List:
    """
    Function creates triangles to select points within.

    Parameters
    ----------
    points : numpy array
        The points to find their neighbors.

    step_size : float
        Lag length.

    angle : float
        The direction of a variogram.

    tolerance : float
        The parameter to control the ratio of triangle base to its height.

    Returns
    -------
    triangles : List
        The list of triangle tuples (three coordinates per polygon and its
        inverted version) ``[triangle, inverted traingle]``

    Notes
    -----
    Each triangle width is equal to ``step_size * tolerance``,
    and baseline point is placed in the middle of triangle's base.
    The height of a triangle is equal to step_size size. Angle points triangle
    to a specific direction on the cartesian plane.
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

    triangles = _prepare_triangle_coordinates(apex=apex,
                                              inv_apex=inv_apex,
                                              base_a=base_a,
                                              base_b=base_b)

    return triangles


def get_triangles_vertices(coordinates: ArrayLike,
                           lags: ArrayLike,
                           direction: float,
                           tolerance: float) -> List:
    """
    Function creates a dictionary with triangle masks coordinates.

    Parameters
    ----------
    coordinates : numpy array
        List of coordinates.

    lags : Ordered Collection
        List of ordered lags.

    direction : float
        The direction of a variogram.

    tolerance : float
        The parameter to control the ratio of triangle base to its height.

    Returns
    -------
    : List
        Masks for each lag, in the same order as lags.
    """
    t_masks = [generate_triangles(coordinates,
                                  h,
                                  direction,
                                  tolerance) for h in lags]
    return t_masks


def select_points_within_triangle(triangle: Tuple,
                                  points: np.ndarray) -> np.ndarray:
    """
    Function selects points inside a triangle defined as its veritces.

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


def select_points_within_ellipse(ellipse_center: np.array,
                                 other_points: np.array,
                                 lag: float,
                                 step_size: float,
                                 w_matrix: np.ndarray) -> np.array:
    """
    Function checks which points from other points are within the point
    range described as an ellipse with center in the point, the semi-major
    axis of the ``step_size`` length, and the semi-minor axis of length
    ``step_size * tolerance``. The direction angle of semi-major axis
    starts from W-E direction, x-axis on the cartesian plane, and goes
    counterclockwise.

    Parameters
    ----------
    ellipse_center : numpy array
        Origin point coordinates.

    other_points : numpy array
        Array with points for which distance is calculated.

    lag : float
        Lag distance.

    step_size : float
        Step size between lags.

    w_matrix : numpy array
        Matrix used for masking values in ellipse.

    Returns
    -------
    : numpy array
        Boolean array of points within ellipse with a center in origin point.
    """

    vector_distance = other_points - ellipse_center

    # Distances
    current_ellipse = _select_ellipse_distances(vector_distance,
                                                w_matrix,
                                                lag,
                                                step_size)

    return current_ellipse


def triangle_mask(triangle_1: Tuple,
                  triangle_2: Tuple,
                  coordinates: np.ndarray) -> np.ndarray:
    """
    Function selects points in the given areas.

    Parameters
    ----------
    triangle_1 : Tuple
        Set of coordinates making triangle: ``([x1, y1], [x2, y2], [x3, y3])``

    triangle_2 : Tuple
        Inverted set of coordinates from ``triangle_1``.

    coordinates : numpy array
        List of coordinates of size Mx2.

    Returns
    -------
    mask : numpy array
        Boolean mask of size Mx2.
    """

    points_in_triangle_a = select_points_within_triangle(
        triangle_1,
        coordinates
    )
    points_in_triangle_b = select_points_within_triangle(
        triangle_2,
        coordinates
    )

    mask = np.logical_or(points_in_triangle_a,
                         points_in_triangle_b)

    return mask


def _calc_angle_between_points(coor1: ArrayLike,
                               coor2: ArrayLike,
                               origin: ArrayLike) -> float:
    """
    Calculates angle between two vectors, both starting in the same point
    but crossing other coordinates.

    Parameters
    ----------
    coor1 : ArrayLike
        Coordinates of the first point.

    coor2 : ArrayLike
        Coordinates of the second point.

    origin : ArrayLike
        Coordinates of origin (usually x=0, y=0)

    Returns
    -------
    : float
        Angle between two vectors.
    """
    ang1 = np.arctan2(coor1[1] - origin[1], coor1[0] - origin[0])
    ang2 = np.arctan2(coor2[:, 1] - origin[1], coor2[:, 0] - origin[0])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def _calc_angle_from_origin(coor: ArrayLike, origin: ArrayLike) -> float:
    """
    Calculates angle between vector starting from origin and crossing point
    and x-axis on the cartesian plane.

    Parameters
    ----------
    coor : ArrayLike
        Coordinates of the first point.

    origin : ArrayLike
        Coordinates of origin (usually x=0, y=0)

    Returns
    -------
    : float
        Angle between line and x-axis.
    """
    ys = coor[:, 1] - origin[1]
    xs = coor[:, 0] - origin[0]
    ang = np.arctan2(ys, xs)
    return np.rad2deg(ang % (2 * np.pi))


def _prepare_triangle_coordinates(apex: ArrayLike,
                                  inv_apex: ArrayLike,
                                  base_a: ArrayLike,
                                  base_b: ArrayLike) -> List:
    """
    Function prepares triangle/mask vertices.

    Parameters
    ----------
    apex : ArrayLike
        Vertex.

    inv_apex : ArrayLike
        Vertex of the inverted triangle.

    base_a : ArrayLike
        First base vertex.

    base_b : ArrayLike
        Second base vertex.

    Returns
    -------
    : List
        Triangular masks.
    """
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


def _rotation_matrix(angle: float) -> np.array:
    """
    Function builds rotation matrix.

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
    """
    Function mutiplies each point from the distances array with
    the weighting matrix to check if point is within elliptical area.

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
    norm_distances = np.matmul(weighting_matrix, distances_array.T).T
    # norm_results = norm_distances.dot(norm_distances.T).diagonal()  # 1.8s
    norm_results = np.einsum('ij, ij -> i',
                             norm_distances,
                             norm_distances)  # 100ms
    norm_results = np.sqrt(norm_results)
    upper_limit = lag
    lower_limit = lag - step_size

    mask_a = norm_results <= upper_limit
    mask_b = norm_results > lower_limit
    mask_c = norm_results != 0

    mask_out = mask_a & mask_b & mask_c

    return mask_out
