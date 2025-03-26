from typing import Union, List, Callable, Tuple

import numpy as np

from pyinterpolate.distance.angular import select_points_within_ellipse, \
    define_whitening_matrix, get_triangles_vertices, \
    build_mask_indices, clean_mask_indices
from pyinterpolate.semivariogram.lags.lags import get_current_and_previous_lag


def from_ellipse(
        fn: Callable,
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        direction: float,
        tolerance: float):
    """
    Function calculates semivariances or
    covariances from elliptical neighborhood.

    Parameters
    ----------
    fn : Callable
        Semivariance or covariance function.

    points : numpy array
        Coordinates and their values: ``[x, y, value]``.

    lags : numpy array
        Bins array.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
        Value in range (0-1] to calculate semi-minor axis length of the
        search area. If tolerance is close to 0 then points must be placed
        in a single line with beginning in the origin of coordinate system
        and direction given by y axis and direction parameter.
            * The major axis length == step_size,
            * The minor axis size == tolerance * step_size.
            * Tolerance == 1 creates the omnidirectional semivariogram.

    Returns
    -------
    output : numpy array
        ``[lag, semivariance | covariance, number of point pairs]``
    """
    values_and_lags = list()

    w_matrix = define_whitening_matrix(theta=direction,
                                       minor_axis_size=tolerance)

    for idx in range(len(lags)):
        value, points_per_lag = _get_values_ellipse(
            fn=fn,
            points=points,
            lags=lags,
            lag_idx=idx,
            w_matrix=w_matrix
        )

        if points_per_lag == 0:
            if idx == 0 and lags[0] == 0:
                values_and_lags.append([0, 0, 0])
            else:
                msg = f'There are no neighbors for a lag {lags[idx]},' \
                      f' the process has been stopped.'
                raise RuntimeError(msg)
        else:
            values_and_lags.append(
                [lags[idx], value, points_per_lag])

    output = np.array(values_and_lags)
    return output


def from_ellipse_cloud(
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        direction: float,
        tolerance: float):
    """
    Function calculates point cloud semivariances from elliptical
    neighborhood.

    Parameters
    ----------
    points : numpy array
        Coordinates and their values: ``[x, y, value]``.

    lags : numpy array
        Bins array.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
        Value in range (0-1] to calculate semi-minor axis length of the
        search area. If tolerance is close to 0 then points must be placed
        in a single line with beginning in the origin of coordinate system
        and direction given by y axis and direction parameter.
            * The major axis length == step_size,
            * The minor axis size == tolerance * step_size.
            * Tolerance == 1 creates the omnidirectional semivariogram.

    Returns
    -------
    output : Dict
        ``{lag: [semivariances]}``
    """
    point_cloud = dict()
    w_matrix = define_whitening_matrix(theta=direction,
                                       minor_axis_size=tolerance)
    for idx in range(len(lags)):
        values = _get_values_ellipse_cloud(
            points=points,
            lags=lags,
            lag_idx=idx,
            w_matrix=w_matrix
        )
        point_cloud[lags[idx]] = values

    return point_cloud


def from_triangle(
        fn: Callable,
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        direction: float,
        tolerance: float
):
    """

    Function selects semivariances or covariances per lag from
    the triangular area.

    Parameters
    ----------
    fn : Callable
        Semivariance or covariance function.

    points : numpy array
        ``[x, y, value]``

    lags : list or numpy array
        The list of lags.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float
        The parameter to control the ratio of triangle base to its height.
    """
    coordinates = points[:, :-1]
    values_and_lags = []

    # Get all triangles instantly
    tr_edges = get_triangles_vertices(
        coordinates=coordinates,
        lags=lags,
        direction=direction,
        tolerance=tolerance
    )

    # Build masks for every lag
    mask_indices = build_mask_indices(
        coordinates=coordinates,
        vertices=tr_edges
    )

    # Clean indices
    mask_indices = clean_mask_indices(
        mask_indices
    )

    for hidx, h in enumerate(lags):
        value, point_pairs = _get_values_triangle(
            fn=fn,
            points=points,
            masks=mask_indices[hidx]
        )

        if point_pairs == 0:
            if hidx == 0:
                values_and_lags.append([h, 0, 0])
            else:
                msg = f'There are no neighbors for a lag {h}, the process ' \
                      f'has been stopped.'
                raise RuntimeError(msg)
        else:
            values_and_lags.append([h, value, point_pairs])

    return np.array(values_and_lags)


def from_triangle_cloud(
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        direction: float,
        tolerance: float
):
    """

    Function selects semivariances per lag from
    the triangular area.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    lags : list or numpy array
        The list of lags.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float
        The parameter to control the ratio of triangle base to its height.
    """
    coordinates = points[:, :-1]
    point_cloud = dict()

    # Get all triangles instantly
    tr_edges = get_triangles_vertices(
        coordinates=coordinates,
        lags=lags,
        direction=direction,
        tolerance=tolerance
    )

    # Build masks for every lag
    mask_indices = build_mask_indices(
        coordinates=coordinates,
        vertices=tr_edges
    )

    # Clean indices
    mask_indices = clean_mask_indices(
        mask_indices
    )

    for hidx, h in enumerate(lags):
        values = _get_values_triangle_cloud(
            points=points,
            masks=mask_indices[hidx]
        )

        point_cloud[h] = values

    return point_cloud


def directional_weighted_semivariance(points: np.array,
                                      lags: np.array,
                                      custom_weights: np.array,
                                      direction: float,
                                      tolerance: float
                                      ) -> np.array:
    """
    Function calculates weighted directional semivariogram.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    lags : numpy array

    custom_weights : optional, Iterable
        Custom weights assigned to points.

    direction : float, optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    Returns
    -------
    : (numpy array)
        [lag, semivariance, number of points within a lag]
    """

    semivariances_and_lags = list()

    w_matrix = define_whitening_matrix(theta=direction,
                                       minor_axis_size=tolerance)

    other_points = points[:, :-1]

    for lag_idx in range(len(lags)):
        weighted_nominator_terms = []
        weighted_denominator_terms = []
        weighted_point_vals = []
        weight_of_points = []

        current_lag, previous_lag = get_current_and_previous_lag(
            lag_idx=lag_idx, lags=lags
        )

        step_size = current_lag - previous_lag

        for idx, point in enumerate(points):
            coordinates = point[:-1]

            mask = select_points_within_ellipse(
                coordinates,
                other_points,
                current_lag,
                step_size,
                w_matrix=w_matrix
            )

            points_in_range = points[mask, -1]

            # Calculate semivariances
            if len(points_in_range) > 0:
                w_vals = custom_weights[mask]
                w_of_single_point = custom_weights[idx]
                w_mult = w_of_single_point * w_vals
                w_sum = w_of_single_point + w_vals
                w_h = w_mult / w_sum
                z_h = (points_in_range - point[-1]) ** 2
                weighted_nominator_terms.extend(z_h)
                weighted_denominator_terms.extend(w_h)
                weighted_point_vals.extend(points_in_range)
                weight_of_points.extend(w_vals)

        if len(weighted_point_vals) == 0:
            if lag_idx == 0:
                semivariances_and_lags.append([current_lag, 0, 0])
            else:
                msg = f'There are no neighbors for a lag {current_lag}, ' \
                      f'the process has been stopped.'
                raise RuntimeError(msg)
        else:
            arr_nom_terms = np.array(weighted_nominator_terms)
            arr_denom_terms = np.array(weighted_denominator_terms)
            arr_vals = np.array(weighted_point_vals)
            arr_weights = np.array(weight_of_points)

            nominator_z = arr_nom_terms - np.average(arr_vals,
                                                     weights=arr_weights)
            nominator_z = np.sum(nominator_z * arr_denom_terms)
            gamma = nominator_z / np.sum(arr_denom_terms)
            average_semivariance = 0.5 * gamma
            semivariances_and_lags.append(
                [current_lag,
                 average_semivariance,
                 len(arr_nom_terms.flatten())]
            )

    output_semivariances = np.array(semivariances_and_lags)
    return output_semivariances


def _get_values_ellipse(
        fn: Callable,
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        lag_idx: int,
        w_matrix: np.ndarray) -> Tuple[float, int]:
    """
    Function selects semivariances or covariances per lag in the ellipse.

    Parameters
    ----------
    fn : Callable
        Semivariance or covariance function.

    points : numpy array
        ``[x, y, value]``

    lags : list or numpy array
        The list of lags.

    lag_idx : int
        Current lag index.

    w_matrix : numpy array
        Matrix used for masking values in ellipse.

    Returns
    -------
    value, point_pairs : [float, int]
        Function value and number of point pairs for a given lag.
    """
    coords = points[:, :-1]

    current_lag, previous_lag = get_current_and_previous_lag(
        lag_idx=lag_idx, lags=lags
    )

    step_size = current_lag - previous_lag

    lag_points = []
    lag_neighbors = []

    for point in points:

        coordinates = point[:-1]

        mask = select_points_within_ellipse(
            ellipse_center=coordinates,
            other_points=coords,
            lag=current_lag,
            step_size=step_size,
            w_matrix=w_matrix
        )

        points_in_range = points[mask, -1]

        # Extend list for calculations
        if len(points_in_range) > 0:
            _pts = [x for x in points_in_range]
            _coo = [point[-1] for _ in range(0, len(_pts))]
            lag_neighbors.extend(_pts)
            lag_points.extend(_coo)

    if len(lag_points) > 0:
        value = fn(
            np.array(lag_points),
            np.array(lag_neighbors)
        )
        return value, len(lag_points)

    return 0.0, 0


def _get_values_ellipse_cloud(
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        lag_idx: int,
        w_matrix: np.ndarray) -> np.ndarray:
    """
    Function selects semivariances or covariances per lag in the ellipse.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    lags : list or numpy array
        The list of lags.

    lag_idx : int
        Current lag index.

    w_matrix : numpy array
        Matrix used for masking values in ellipse.

    Returns
    -------
    values : numpy array
        Semivariances for a given lag.
    """
    coords = points[:, :-1]

    current_lag, previous_lag = get_current_and_previous_lag(
        lag_idx=lag_idx, lags=lags
    )

    step_size = current_lag - previous_lag

    lag_points = []
    lag_neighbors = []

    for point in points:

        coordinates = point[:-1]

        mask = select_points_within_ellipse(
            ellipse_center=coordinates,
            other_points=coords,
            lag=current_lag,
            step_size=step_size,
            w_matrix=w_matrix
        )

        points_in_range = points[mask, -1]

        # Extend list for calculations
        if len(points_in_range) > 0:
            _pts = [x for x in points_in_range]
            _coo = [point[-1] for _ in range(0, len(_pts))]
            lag_neighbors.extend(_pts)
            lag_points.extend(_coo)

    if len(lag_points) > 0:
        values = np.square(
            np.array(lag_points) - np.array(lag_neighbors)
        )
        return values

    return np.array([])


def _get_values_triangle(
        fn: Callable,
        points: np.ndarray,
        masks: np.ndarray) -> Tuple[float, int]:
    """
    Function selects semivariances or covariances per lag in the triangle.

    Parameters
    ----------
    fn : Callable
        Semivariance or covariance function.

    points : numpy array
        ``[x, y, value]``

    masks : numpy array
        Masks for each point and its neighbors.

    Returns
    -------
    value, point_pairs : [float, int]
        Function value and number of point pairs for a given lag.
    """
    lag_points = []
    lag_neighbors = []

    for idx, point in enumerate(points):
        mask = masks[idx]
        points_in_range = points[mask]

        # Calculate semivariances
        if len(points_in_range) > 0:
            _pts = [x[-1] for x in points_in_range]
            _coo = [point[-1] for _ in range(0, len(_pts))]
            lag_neighbors.extend(_pts)
            lag_points.extend(_coo)

    if len(lag_points) > 0:
        value = fn(
            np.array(lag_points),
            np.array(lag_neighbors)
        )
        return value, len(lag_points)

    return 0.0, 0


def _get_values_triangle_cloud(
        points: np.ndarray,
        masks: np.ndarray) -> np.ndarray:
    """
    Function selects semivariances or covariances per lag in the triangle.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    masks : numpy array
        Masks for each point and its neighbors.

    Returns
    -------
    value, point_pairs : [float, int]
        Function value and number of point pairs for a given lag.
    """
    lag_points = []
    lag_neighbors = []

    for idx, point in enumerate(points):
        mask = masks[idx]
        points_in_range = points[mask]

        # Calculate semivariances
        if len(points_in_range) > 0:
            _pts = [x[-1] for x in points_in_range]
            _coo = [point[-1] for _ in range(0, len(_pts))]
            lag_neighbors.extend(_pts)
            lag_points.extend(_coo)

    if len(lag_points) > 0:
        values = np.square(
            np.array(lag_points) - np.array(lag_neighbors)
        )
        return values

    return np.array([])
